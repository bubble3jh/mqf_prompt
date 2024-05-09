import pytorch_lightning as pl
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import csv

def hook_fn(module, input, output):
    global hidden_output
    hidden_output = output

def normalizer(x, x_prompted):
    x_max = x.max(dim=-1, keepdim=True)[0]; x_min = x.min(dim=-1, keepdim=True)[0] # 256, 1, 1
    x_prompted_max = x_prompted.max(dim=-1, keepdim=True)[0]; x_prompted_min = x_prompted.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    kk = scale*(x_prompted - x_prompted_min) + x_min
    return kk

def normalize_keys(keys):
    # Min-Max Scaling을 사용한 정규화
    min_val = keys.min(dim=-1, keepdim=True)[0]
    max_val = keys.max(dim=-1, keepdim=True)[0]
    normalized_keys = (keys - min_val) / (max_val - min_val)
    return normalized_keys

def global_normalizer(prompt, x_min, x_max):
    x_prompted_max = prompt.max(dim=-1, keepdim=True)[0]; x_prompted_min = prompt.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    norm_prompt = scale*(prompt - x_prompted_min) + x_min
    return norm_prompt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        return x
    
class PromptCNN(nn.Module):
    def __init__(self):
        super(PromptCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.to(self.conv1d.weight.device)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.squeeze(1)
        return x
    
class SimpleLinear(nn.Module):
    def __init__(self, in_size):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(in_size, in_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return x

    
class L2Prompt(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.model_config = model_config        
        self.k = config.k
        self.num_pool = config.num_pool
        self.penalty = config.penalty
        self.cnn_layer = SimpleCNN()
        self.prompt_cnn = PromptCNN()
        #self.linear_layer = SimpleLinear(model_config["data_dim"])
        
        if self.config.fixed_key:
            self.keys = torch.randn(self.num_pool,self.model_config["data_dim"]) #(10, 625)
            self.keys = self.keys.cuda()
        else:
            self.keys = nn.Parameter(torch.randn(self.num_pool,self.model_config["data_dim"])) #(10, 625)
        
        if self.config.fixed_prompt:
            self.prompt = torch.randn(self.num_pool , self.model_config["data_dim"])
            self.prompt = self.prompt.cuda()
        else:    
            self.prompt = nn.Parameter(torch.randn(self.num_pool , self.model_config["data_dim"]))
        self.frequency_dict = None 
        
    def count_frequencies(self, tensor,k): 
        unique_values, counts = torch.unique(tensor, return_counts=True)
        frequencies = {}
        for i in range(20):
            frequencies[i] = 0
        for value, count in zip(unique_values, counts):
            frequencies[value.item()] = count.item()
        return frequencies   
   
    def forward(self, x, mode): 
        
        bz = x['ppg'].shape[0]   
        
        if self.config.exp.data_name == 'bcg' and self.config.cnn:
            x['ppg'] = self.cnn_layer(x['ppg'])
        prompt_cnn = self.prompt_cnn(self.prompt)    
        score_ = 1 - F.cosine_similarity(x['ppg'], self.keys, dim=-1) #cos distance        
        score, idx = torch.topk(score_,self.k,largest=False, axis=-1)
        idx = idx.squeeze()
        probs = F.softmax(score_, dim = -1)
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()      
        
        if self.config.cnn:
            if self.k != 1:
                if bz == 1:
                    prompt = prompt_cnn[idx, :].mean(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                else:
                    
                    prompt = prompt_cnn[idx, :].mean(dim=1).unsqueeze(dim=1)
            else:
                if bz == 1:
                    prompt = prompt_cnn[idx.long(), :].unsqueeze(dim=0).unsqueeze(dim=0)
                else:    
                    prompt = prompt_cnn[idx.long(), :].unsqueeze(dim=1)
        
        else:
            if self.k != 1:
                if bz == 1:
                    prompt = self.prompt[idx, :].mean(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
                else:
                    
                    prompt = self.prompt[idx, :].mean(dim=1).unsqueeze(dim=1)
            else:
                if bz == 1:
                    prompt = self.prompt[idx.long(), :].unsqueeze(dim=0).unsqueeze(dim=0)
                else:    
                    prompt = self.prompt[idx.long(), :].unsqueeze(dim=1)
            
        
        self.frequency_dict = [self.count_frequencies(idx, self.k)]        
        #with open(f'/mlainas/yewon/bp-benchmark/freq_{self.config.penalty}_{mode}.csv', 'a', newline='') as csvfile:
        #    writer = csv.writer(csvfile)
        #    # 데이터 쓰기
        #    for row in [self.frequency_dict]:
        #        writer.writerow(row)
                        
        if self.config.glonorm:
            prompt = global_normalizer(prompt, self.x_min, self.x_max)
        
        if self.config.mul:
            prompted = x['ppg']*self.config.global_coeff*prompt
        else:    
            prompted = x['ppg'] + self.config.global_coeff*prompt
        '''
        if mode == 'test':
            b = prompt.cpu().numpy()
            c = prompted.cpu().numpy()
            d = x['ppg'].cpu().numpy()
            import numpy as np
            np.save(f'/mlainas/yewon/bp-benchmark/prompt_{self.config.penalty}_{mode}.npy', b)
            np.save(f'/mlainas/yewon/bp-benchmark/prompted_{self.config.penalty}_{mode}.npy', c)
            np.save(f'/mlainas/yewon/bp-benchmark/data_{self.config.penalty}_{mode}.npy', d)
       '''     
        
        torch.save(prompt, 'prompt.pt')
        
        score = score.mean()
        
        if self.config.fixed_key:
            score = 0.          
        
        return prompted, score, entropy
    
class Custom_model(pl.LightningModule):
    def __init__(self, model, data_shape, model_config, config, stats):
        super().__init__()
        self.config = config
        self.res_model = model
        self.data_shape = data_shape
        self.model_config = model_config
        self.ppg_min = stats[0]
        self.ppg_max = stats[1]
        self.prompt_learner_glo =L2Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)

        #Loss Function
        if self.config.group_avg:
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.MSELoss()

    def hook_fn(self, module, input, output):
        self.hidden_output = output

    def _shared_step(self, batch, mode):
        if self.config.group_avg:
            x_ppg, y, group, x_abp, peakmask, vlymask = batch
        else:
            x_ppg, y, x_abp, peakmask, vlymask = batch
        
        merged, l2pscore, entropy = self.prompt_learner_glo(x_ppg, mode)

        if self.config.normalize:
            merged = normalizer(x_ppg["ppg"], merged)
        if self.config.clip:
            merged = torch.clamp(merged, min= self.ppg_min,max=self.ppg_max)
        torch.save(merged, "merged_1.pt")
        pred = self.res_model(merged)
               
        if self.config.group_avg:
            losses = self.criterion(pred, y)
            loss = self.grouping(losses, group)
            if self.config.method == "prompt_global":
                loss = loss + self.config.score_ratio*l2pscore 
                if self.config.penalty:
                    loss = loss - self.config.lamb*entropy
                return loss, pred, x_abp, y, group
            return loss, pred, x_abp, y, group

        else:
            loss = self.criterion(pred, y)
            if self.config.method == "prompt_global":
                loss = loss + self.config.score_ratio*l2pscore #- entropy
                if self.config.penalty:
                    loss = loss - self.config.lamb*entropy
                    return loss, pred, x_abp, y
            return loss, pred, x_abp, y
        
    def grouping(self, losses, group):
        group_type = torch.arange(0,4).cuda()
        group_map = (group_type.view(-1,1)==group).float()
        group_count = group_map.sum(1)
        group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
        group_loss = group_loss_map.sum(1)                         # (4,2)

        # Average only across the existing group
        mask = group_count != 0
        avg_per_group = torch.zeros_like(group_loss)
        avg_per_group[mask, :] = group_loss[mask, :] / group_count[mask].unsqueeze(1)
        exist_group = mask.sum()
        avg_group = avg_per_group.sum(0)/exist_group
        loss = avg_group.sum()/2
        return loss

    def training_step(self, batch, batch_idx):
        if self.config.group_avg:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode = 'train')
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch,  mode = 'train')
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}    
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"] for v in train_step_outputs], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        if self.config.group_avg:
            loss, pred_bp, t_abp, label, group  = self._shared_step(batch, mode='val')
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch, mode='val')        
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"] for v in val_step_end_out], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        if self.config.group_avg:
            loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode='test')  
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
            self.log('test_loss', loss, prog_bar=True)
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch, mode='test')   
            self.log('test_loss', loss, prog_bar=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"] for v in test_step_end_out], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out
    
    def _cal_metric(self, logit: torch.tensor, label: torch.tensor, group=None):
        prev_mse = (logit-label)**2
        prev_mae = torch.abs(logit-label)
        prev_me = logit-label
        mse = torch.mean(prev_mse)
        mae = torch.mean(prev_mae)
        me = torch.mean(prev_me)
        std = torch.std(torch.mean(logit-label, dim=1))
        if self.config.group_avg:
            group_mse = self.grouping(prev_mse, group)
            group_mae = self.grouping(prev_mae, group)
            group_me = self.grouping(prev_me, group)
            return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me} 
        else:
            return {"mse":mse, "mae":mae, "std": std, "me": me} 
    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        return optimizer
