import pytorch_lightning as pl
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F

def hook_fn(module, input, output):
    global hidden_output
    hidden_output = output

def normalizer(x, x_prompted):
    x_max = x.max(dim=-1, keepdim=True)[0]; x_min = x.min(dim=-1, keepdim=True)[0] # 256, 1, 1
    x_prompted_max = x_prompted.max(dim=-1, keepdim=True)[0]; x_prompted_min = x_prompted.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    kk = scale*(x_prompted - x_prompted_min) + x_min
    return kk

def global_normalizer(prompt, x_min, x_max):
    x_prompted_max = prompt.max(dim=-1, keepdim=True)[0]; x_prompted_min = prompt.min(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
    norm_prompt = scale*(prompt - x_prompted_min) + x_min
    return norm_prompt

class Prompt(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.model_config = model_config
        self.prompt = nn.Parameter(torch.randn(1, self.model_config["data_dim"]))

    def forward(self, x):
        bz = x['ppg'].shape[0]
        expanded_tensor = torch.unsqueeze(self.prompt, 0)
        prompt = expanded_tensor.expand(bz, -1, -1)  # 256, 1, 256
        if self.config.glonorm:
            prompt = global_normalizer(prompt, self.x_min, self.x_max)
        prompted = x['ppg'] + self.config.global_coeff*prompt
        torch.save(prompt, 'prompt.pt')
        return prompted



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
        self.prompt = nn.Parameter(torch.randn(self.num_pool , self.model_config["data_dim"]))
        self.keys = nn.Parameter(torch.randn(self.num_pool,self.model_config["data_dim"])) #(10, 625)
        
    def count_frequencies(self, tensor): 
        unique_values, counts = torch.unique(tensor, return_counts=True)
        frequencies = {}
        for i in range(10):
            frequencies[i] = 0
        for value, count in zip(unique_values, counts):
            frequencies[value.item()] = count.item()
        return frequencies   
   
    def forward(self, x):

        bz = x['ppg'].shape[0]
        #x['ppg'] (256,1,625) 
        score_ = 1 - F.cosine_similarity(x['ppg'], self.keys, dim=-1) # cos distance
        score, idx = torch.topk(score_,self.k,largest=False, axis=-1)
        idx = idx.squeeze()
        probs = F.softmax(score_, dim = -1)
        entropy = -torch.sum(probs * torch.log(probs))
        expanded_tensor = torch.unsqueeze(self.prompt, 0)
        expanded_prompt = expanded_tensor.expand(bz, -1, -1)
        mask = torch.ones_like(expanded_prompt)
        m_ = torch.ones_like(score_)
        if bz == 1:
            mask[:,idx,:] = 0
            m_[:,idx] = 0
        else:
            for i in range(bz):
                mask[i,idx[i],:] = 0
                m_[i,idx[i]] = 0
        prompt_ = self.prompt[idx.long(), :].unsqueeze(dim=1) 
        mask = 1-mask
        m_ = 1-m_
        score = score_*m_
        prompt = (self.prompt*mask).sum(dim=1).unsqueeze(1) 
        if self.config.glonorm:
            prompt = global_normalizer(prompt, self.x_min, self.x_max)
        prompted = x['ppg'] + self.config.global_coeff*prompt
        torch.save(prompt, 'prompt.pt')
        return prompted, score.sum(), entropy
    
class PromptGEN(nn.Module):
    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, self.model_config["data_dim"])
        self.relu = nn.ReLU()
        #self.prompt = nn.Parameter(0.1*torch.rand(1,self.prompt_init))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.unsqueeze(1)
        return x


class PromptGEN_Deconv(nn.Module):
    def __init__(self, config, model_config):
        super(PromptGEN_Deconv, self).__init__()

        self.config = config
        self.model_config = model_config
        # Fully connected layer to reshape the input
        self.fc = nn.Linear(self.model_config["last_dim"], 32 * 40)  # 32 channels, length 40

        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape the input using the fully connected layer
        x = self.fc(x)
        x = x.view(-1, 32, 40)  # Reshape to (bs, 32, 40)

        # Apply deconvolution layers
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv2(x)

        # Resize to match the desired output shape (bs, 1, self.model_config['data_dim'])
        x = nn.functional.interpolate(x, size=self.model_config['data_dim'])
        # print(self.config.param_loader.SP_min)
        # print(self.config.param_loader.SP_max)

        # print(self.config.param_loader.DP_min)
        # print(self.config.param_loader.DP_max)
        return x

class PromptGEN_Deconv_0(nn.Module):
    def __init__(self, config, model_config):
        super(PromptGEN_Deconv_0, self).__init__()

        self.config = config
        self.model_config = model_config
        # Define the deconvolution layers

        self.deconv1 = nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv2(x)

        x = nn.functional.interpolate(x, size=self.model_config['data_dim'])
        return x


class PromptGEN_Deconv_1(nn.Module):
    def __init__(self, config, model_config):
        super(PromptGEN_Deconv_1, self).__init__()

        self.config = config
        self.model_config = model_config
        # Define the deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv2(x)

        x = nn.functional.interpolate(x, size=self.model_config['data_dim'])

        return x

class PromptGEN_Deconv_2(nn.Module):
    def __init__(self,config, model_config):
        super(PromptGEN_Deconv_2, self).__init__()

        self.config = config
        self.model_config = model_config
        # Define the deconvolution layers
        self.deconv1 = nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(128, 1, kernel_size=4, stride=2, padding=1)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv2(x)

        # Resize to match the desired output shape (bs, 1, self.model_config['data_dim'])
        x = nn.functional.interpolate(x, size=self.model_config['data_dim'])

        return x


class Custom_model(pl.LightningModule):
    def __init__(self, model, data_shape, model_config, config, stats):
        super().__init__()
        self.config = config
        self.res_model = model
        self.data_shape = data_shape
        self.model_config = model_config
        self.ppg_min = stats[0]
        self.ppg_max = stats[1]
        if self.config.method == "prompt_gen":
            self.extractor = copy.deepcopy(self.res_model)
            self.extractor.model.main_clf = nn.Identity()
            self.prompt_learner_gen = PromptGEN_Deconv(self.config, self.model_config)

        elif self.config.method == "prompt_lowgen":
            self.extractor = copy.deepcopy(self.res_model)
            self.hidden_output = None
            self.handle = self.extractor.model.basicblock_list[self.config.layer_num].register_forward_hook(self.hook_fn)
            if self.config.layer_num == 0:
                self.prompt_learner_gen = PromptGEN_Deconv_0(self.config, self.model_config)
            elif self.config.layer_num == 1:
                self.prompt_learner_gen = PromptGEN_Deconv_1(self.config, self.model_config)
            elif self.config.layer_num == 2:
                self.prompt_learner_gen = PromptGEN_Deconv_2(self.config, self.model_config)

        elif self.config.method == "prompt_global":
            #self.prompt_learner_glo = Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)
            self.prompt_learner_glo =L2Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)

        elif self.config.method == "prompt_glogen":
            self.extractor = copy.deepcopy(model)
            self.extractor.model.main_clf = nn.Identity()
            self.prompt_learner_gen = PromptGEN_Deconv(self.config, self.model_config)
            self.prompt_learner_glo = Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)

        elif self.config.method == "prompt_lowglogen":
            self.extractor = copy.deepcopy(self.res_model)
            self.hidden_output = None
            self.handle = self.extractor.model.basicblock_list[self.config.layer_num].register_forward_hook(
                self.hook_fn)
            if self.config.layer_num == 0:
                self.prompt_learner_gen = PromptGEN_Deconv_0(self.config, self.model_config)
            elif self.config.layer_num == 1:
                self.prompt_learner_gen = PromptGEN_Deconv_1(self.config, self.model_config)
            elif self.config.layer_num == 2:
                self.prompt_learner_gen = PromptGEN_Deconv_2(self.config, self.model_config)
            self.prompt_learner_glo = Prompt(self.config, self.model_config, self.ppg_min, self.ppg_max)

        #Loss Function
        if self.config.group_avg:
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.MSELoss()

    def hook_fn(self, module, input, output):
        self.hidden_output = output

    def _shared_step(self, batch):
        if self.config.group_avg:
            x_ppg, y, group, x_abp, peakmask, vlymask = batch
        else:
            x_ppg, y, x_abp, peakmask, vlymask = batch

        if self.config.method == "prompt_gen":
            hidden = self.extractor(x_ppg["ppg"])
            prompt = self.prompt_learner_gen(hidden)
            #torch.save(x_ppg, 'x.pt')
            #torch.save(y, 'y.pt')
            #torch.save(prompt, 'gen.pt')
            merged = x_ppg["ppg"] + self.config.gen_coeff *  prompt

        if self.config.method == "prompt_lowgen":
            hidden = self.extractor(x_ppg["ppg"])
            prompt = self.prompt_learner_gen(self.hidden_output)
            merged = x_ppg["ppg"] + self.config.gen_coeff * prompt

        if self.config.method == "prompt_global":
            merged, l2pscore, entropy = self.prompt_learner_glo(x_ppg)
      
        if self.config.method == "prompt_glogen":
            hidden = self.extractor(x_ppg["ppg"])
            gen_prompt = self.prompt_learner_gen(hidden)
            prompted = self.prompt_learner_glo(x_ppg)
            #merged = self.config.global_coeff*prompted + gen_prompt
            torch.save(x_ppg, 'x_1.pt')
            torch.save(y, 'y_1.pt')
            torch.save(gen_prompt, 'gen_1.pt')
            merged = prompted + self.config.gen_coeff*gen_prompt

        if self.config.method == "prompt_lowglogen":
            hidden = self.extractor(x_ppg["ppg"])
            gen_prompt = self.prompt_learner_gen(self.hidden_output)
            prompted = self.prompt_learner_glo(x_ppg)
            # merged = self.config.global_coeff*prompted + gen_prompt
            merged = prompted + self.config.gen_coeff * gen_prompt

        if self.config.normalize:
            merged = normalizer(x_ppg["ppg"], merged)
        if self.config.clip:
            merged = torch.clamp(merged, min= self.ppg_min,max=self.ppg_max)
        torch.save(merged, "merged_1.pt")

        pred = self.res_model(merged)
        # print(y[0])
        # print(pred[0])
        # print(x_ppg['ppg'][0])
        # print(gen_prompt[0])
        # print(self.prompt_learner_glo.prompt)
        # print(merged[0])
        ######################
        if self.config.group_avg:
            losses = self.criterion(pred, y)
            loss = self.grouping(losses, group)
            # group_type = torch.arange(0,4).cuda()
            # group_map = (group_type.view(-1,1)==group).float()
            # group_count = group_map.sum(1)
            # group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
            # group_loss = group_loss_map.sum(1)                          # (4,2)

            # # Average only across the existing group
            # mask = group_count != 0
            # avg_per_group = torch.zeros_like(group_loss)
            # avg_per_group[mask, :] = group_loss[mask, :] / group_count[mask].unsqueeze(1)
            # exist_group = mask.sum()
            # avg_group = avg_per_group.sum(0)/exist_group
            # loss = avg_group.sum()/2
            if self.config.method == "prompt_global":
                loss = loss + self.config.score_ratio*l2pscore 
                
                if self.config.penalty:
                    loss = loss - entropy
                return loss, pred, x_abp, y, group
            return loss, pred, x_abp, y, group

        else:
            loss = self.criterion(pred, y)
            if self.config.method == "prompt_global":
                loss = loss + self.config.score_ratio*l2pscore #- entropy
                if self.config.penalty:
                    print(entropy)
                    loss = loss - self.config.lamb*entropy
                return loss, pred, x_abp, y
            return loss, pred, x_abp, y
        ######################

        
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
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch)
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
            loss, pred_bp, t_abp, label, group  = self._shared_step(batch)
            self.log('val_loss', loss, prog_bar=True, on_epoch=True)
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch)        
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
            loss, pred_bp, t_abp, label, group = self._shared_step(batch)  
            return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  
            self.log('test_loss', loss, prog_bar=True)
        else:
            loss, pred_bp, t_abp, label = self._shared_step(batch)   
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
        if self.config.method == "prompt_global":
            optimizer = torch.optim.Adam([
                {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        if (self.config.method == "prompt_gen") or (self.config.method == "prompt_lowgen"):
            optimizer = torch.optim.Adam([
                {'params': self.prompt_learner_gen.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        if (self.config.method == "prompt_glogen") or (self.config.method == "prompt_lowglogen"):
            optimizer = torch.optim.Adam([
                {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},
                {'params': self.prompt_learner_gen.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd}])
        return optimizer
