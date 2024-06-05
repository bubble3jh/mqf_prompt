import pytorch_lightning as pl
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
import csv
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from pyts.decomposition import SingularSpectrumAnalysis
import pywt
from core.utils import perform_pca, project_to_pca_plane, loc_z
import wandb

def hook_fn(module, input, output):
    global hidden_output
    hidden_output = output

# def normalizer(x, x_prompted):
#     x_max = x.max(dim=-1, keepdim=True)[0]; x_min = x.min(dim=-1, keepdim=True)[0] # 256, 1, 1
#     x_prompted_max = x_prompted.max(dim=-1, keepdim=True)[0]; x_prompted_min = x_prompted.min(dim=-1, keepdim=True)[0]
#     scale = (x_max - x_min) / (x_prompted_max - x_prompted_min)
#     kk = scale*(x_prompted - x_prompted_min) + x_min
#     return kk

def normalizer(x, x_prompted):
    x_mean = x.mean(dim=-1, keepdim=True)  
    x_std = x.std(dim=-1, keepdim=True) + 1e-6  
    normalized_x_prompted = (x_prompted - x_mean) / x_std
    return normalized_x_prompted

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

class PPGEmbeddingGenerator(nn.Module):  
    def __init__(self, use_group, group_embedding_dim=64, num_groups=4):
        super(PPGEmbeddingGenerator, self).__init__()  # Initialize the parent class
        if use_group:
            self.use_group = use_group
            self.group_embedding_dim = group_embedding_dim
            self.group_lookup_table = nn.Embedding(num_groups, group_embedding_dim) if num_groups is not None else None
        
    # def pca_transform(self, data, num_components=64):
    #     reshaped_data = data.view(data.size(0), -1).cpu().numpy()
    #     pca = PCA(n_components=min(num_components, reshaped_data.shape[1]))
    #     transformed_data = pca.fit_transform(reshaped_data)
    #     return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def pca_transform(self, data, pca_matrix=None, pca_mean=None):
        transformed_data = project_to_pca_plane(data, pca_matrix, pca_mean)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def ssa_transform(self, data, window_size=125, groups=20):
        reshaped_data = data.view(data.size(0), -1).cpu().numpy()
        ssa = SingularSpectrumAnalysis(window_size=min(window_size, reshaped_data.shape[1]//2), groups=min(groups, reshaped_data.shape[1]//5))
        transformed_data = ssa.fit_transform(reshaped_data)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def wavelet_transform(self, data, wavelet_name='db1', level=5):
        transformed_data = []
        for batch in data:
            coeffs = pywt.wavedec(batch.cpu().numpy().flatten(), wavelet_name, level=min(level, int(np.log2(batch.numel()))-1))
            flat_coeffs = np.concatenate(coeffs)
            # transformed_data.append(flat_coeffs[:num_components])
            transformed_data.append(flat_coeffs)
        transformed_data = np.array(transformed_data)
        return torch.tensor(transformed_data, dtype=torch.float32, device=data.device)

    def fft_transform(self, data):
        num_components = (data.size(2) // 2) + 1 # FFT has symentic matrix
        
        reshaped_data = data.view(data.size(0), -1).cpu().numpy()
        transformed_data = np.fft.fft(reshaped_data, axis=1)
        return torch.tensor(np.real(transformed_data[:, :num_components]), dtype=torch.float32, device=data.device)
    
    def gen_ppg_emb(self, ppg, groups, pca_matrix, pca_mean):
        # Ensure the input data is in the correct shape
        assert len(ppg.shape) == 3 and ppg.shape[1] == 1, "PPG data should be in shape [batch_size, 1, seq_len]"
        
        device = ppg.device  # Get the device of the input PPG data
        
        # Generate Group embedding
        # group_emb = self.group_lookup_table(groups.to(device))  # Move groups to the same device as ppg
        
        # Generate PCA embedding
        pca_emb = self.pca_transform(ppg, pca_matrix, pca_mean)
        
        # Generate FFT embedding
        fft_emb = self.fft_transform(ppg)
        
        # Generate Wavelet embedding
        wavelet_emb = self.wavelet_transform(ppg)

        return pca_emb, fft_emb, wavelet_emb
    
class L2Prompt(nn.Module):
    def __init__(self, config, model_config, x_min, x_max):
        super().__init__()
        self.config = config
        self.x_min = x_min
        self.x_max = x_max
        self.model_config = model_config   
        # self.k = config.k
        self.num_pool = config.num_pool
        self.penalty = config.penalty
        self.top_k = 1 # select top - 1
        self.ppg_embedding_generator = PPGEmbeddingGenerator(config.use_group, self.model_config["emb_dim"])
            
        # Projection Matrix for feature querys
        self.pca_proj = nn.Linear(config.pca_dim, model_config["emb_dim"], bias=False)
        self.fft_proj = nn.Linear(model_config["data_dim"]//2 + 1, model_config["emb_dim"], bias=False)
        self.wavelet_proj = nn.Linear(model_config["wavelet_dim"], model_config["emb_dim"], bias=False)    
        
        # Initialize learnable parameters for keys and prompts
        self.keys = nn.Parameter(torch.randn(self.num_pool, 3 if not config.use_group else 4, self.model_config["emb_dim"]))
        nn.init.xavier_uniform_(self.keys)

        self.prompts = nn.Parameter(torch.randn(self.num_pool, 1, self.model_config["data_dim"]))
        nn.init.xavier_uniform_(self.prompts)
    
        # Initialize learnable weights
        self.weight_per_prompt = config.weight_per_prompt
        if self.weight_per_prompt:
            self.learnable_weights = nn.Parameter(self._initialize_weights(3 if not config.use_group else 4, self.num_pool))
        else:
            self.learnable_weights = nn.Parameter(self._initialize_weights(3 if not config.use_group else 4))
    
    def _initialize_weights(self, size, num_pool=None):
        if num_pool:
            weights = torch.empty((size, num_pool))
            torch.nn.init.xavier_uniform_(weights)
        else:
            weights = torch.empty((size,1))
            torch.nn.init.xavier_uniform_(weights)
            weights = weights.squeeze()
        return weights

    def count_frequencies(self, tensor,k): 
        unique_values, counts = torch.unique(tensor, return_counts=True)
        frequencies = {}
        for i in range(20):
            frequencies[i] = 0
        for value, count in zip(unique_values, counts):
            frequencies[value.item()] = count.item()
        return frequencies   
    
    def forward(self, x, group_labels, pca_matrix, pca_mean):
        bz = x['ppg'].shape[0]
        
        # Generate PPG embeddings
        pca_emb, fft_emb, wavelet_emb = self.ppg_embedding_generator.gen_ppg_emb(x['ppg'], group_labels, pca_matrix, pca_mean)

        if len(pca_emb.shape) == 1:
            pca_emb = pca_emb.unsqueeze(0)
        if len(fft_emb.shape) == 1:
            fft_emb = fft_emb.unsqueeze(0)
        if len(wavelet_emb.shape) == 1:
            wavelet_emb = wavelet_emb.unsqueeze(0)

        # Project each feature to 64 dimensions
        pca_emb = self.pca_proj(pca_emb)
        fft_emb = self.fft_proj(fft_emb)
        wavelet_emb = self.wavelet_proj(wavelet_emb)

        # Concatenate the projected features to form the query
        queries = torch.stack([pca_emb, fft_emb, wavelet_emb], dim=1)
        
        d_k = queries.size(-1)  # query의 마지막 차원의 크기
        
        cos_sim = torch.einsum('bqd,nqd->bqn', queries, self.keys) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        gumbel_sample = F.gumbel_softmax(cos_sim, tau=1.0, hard=True)

        top1_prompts = torch.einsum('bki,nd->bkd', gumbel_sample, self.prompts.squeeze(1))

        # top1_indices = gumbel_sample.argmax(dim=-1).clone().detach().to(torch.int64)
        top1_indices = gumbel_sample.argmax(dim=-1).to(torch.int64)
        if self.config.prompt_weights == 'cos_sim':
            # Compute matching scores and apply softmax to get weights
            matching_scores = cos_sim.gather(-1, top1_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, 3)
            weights = F.softmax(matching_scores, dim=-1)  # Shape: (batch_size, 3)
        elif self.config.prompt_weights == 'learnable':
            if not self.weight_per_prompt:
                weights = self.learnable_weights.expand([bz, self.learnable_weights.shape[0]])
                weights = F.softmax(weights, dim=-1)
            else:
                # Use learnable weights
                weights = torch.stack([
                    self.learnable_weights[i].gather(0, top1_indices[:, i]) for i in range(3)
                ], dim=1)  # Shape: (batch_size, 3)
                weights = F.softmax(weights, dim=-1)  # Shape: (batch_size, 3)
        
        # Compute weighted sum of prompts
        final_prompt = (weights.unsqueeze(-1) * top1_prompts).sum(dim=1, keepdim=True)  # Shape: (batch_size, 1, prompt_dim)
                
        if self.config.glonorm:
            final_prompt = global_normalizer(final_prompt, self.x_min, self.x_max)
        
        if self.config.mul:
            prompted_signal = x['ppg']*self.config.global_coeff*final_prompt
        else:
            prompted_signal = x['ppg'] + self.config.global_coeff*final_prompt
        
        # Calculate pull_constraint loss (similarity loss) using cos_sim
        sim_pull = cos_sim.gather(-1, top1_indices.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, 4)
        # sim_pull = cos_sim.gather(-1, top1_indices).squeeze(-1)
        sim_loss = torch.clamp(1 - sim_pull.mean(), min=0)
          # Negative to maximize similarity

        # top_prompt_idx = top1_indices.detach().cpu().numpy()
        # wandb.log({"top_prompt_idx": wandb.Table(data=top_prompt_idx, columns=["PCA", "FFT", "Wave"])})

        # Calculate entropy penalty to ensure diverse prompt selection
        entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()  # Shape: scalar
        entropy_penalty = -entropy  # Negative to minimize entropy
        
        return prompted_signal, sim_loss, entropy_penalty
    
class Custom_model(pl.LightningModule):
    def __init__(self, model, data_shape, model_config, config, stats, fold):
        super().__init__()
        self.fold = fold
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
        print('init model')
        
        self.pca_matrix = None
        self.pca_train_mean = 0

    def hook_fn(self, module, input, output):
        self.hidden_output = output

    def _shared_step(self, batch, mode):
        x_ppg, y, group, x_abp, peakmask, vlymask = batch
        if (self.pca_matrix == None) & (self.step_mode=="val"):
            merged, sim_loss, entropy_penalty = self.prompt_learner_glo(x_ppg, group, self.sanity_pca_matrix, self.sanity_val_mean)
        else:
            merged, sim_loss, entropy_penalty = self.prompt_learner_glo(x_ppg, group, self.pca_matrix, self.pca_train_mean)
        if self.config.normalize:
            # merged = normalizer(x_ppg["ppg"], merged)
            merged = loc_z(merged, self.config)
        if self.config.clip:
            merged = torch.clamp(merged, min= self.ppg_min,max=self.ppg_max)
        import pdb;pdb.set_trace()
        # torch.save(merged, "merged_1.pt")
        pred = self.res_model(merged)
               
        if self.config.group_avg:
            raise ValueError("We do not use group loss")
            losses = self.criterion(pred, y)
            loss = self.grouping(losses, group)
            if self.config.method == "prompt_global":
                loss = loss + self.config.qk_sim_coeff*sim_loss 
                if self.config.penalty:
                    loss = loss + self.config.penalty_scaler*entropy_penalty
                return loss, pred, x_abp, y, group
            return loss, pred, x_abp, y, group

        else:
            loss = self.criterion(pred, y)
            if not self.config.ignore_wandb:
                wandb.log(
                    {f'Fold{self.fold}/{mode}_reg_loss':loss,
                        f'Fold{self.fold}/{mode}_qk_sim_loss':sim_loss,
                        f'Fold{self.fold}/{mode}_penalty_loss':entropy_penalty,
                        f'Fold{self.fold}/{mode}_total_loss': loss + self.config.qk_sim_coeff*sim_loss + self.config.penalty_scaler*entropy_penalty,
                        'epoch': self.current_epoch}
                        )
            if self.config.method == "prompt_global":
                loss = loss + self.config.qk_sim_coeff*sim_loss #- entropy
                if self.config.penalty:
                    loss = loss + self.config.penalty_scaler*entropy_penalty
            
            return loss, pred, x_abp, y, group
        
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
        self.step_mode = 'train'
        if (self.pca_matrix==None):
            assert len(batch[0]['ppg']==self.config.param_model.batch_size)
            self.pca_matrix, self.pca_train_mean = perform_pca(batch[0]['ppg'], n_components=self.config.pca_dim)
        loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode = 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group} 
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        group = torch.cat([v["group"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        self.step_mode = 'val'
        if (self.pca_matrix == None):
            self.sanity_pca_matrix = torch.randn((batch[0]['ppg'].shape[-1], self.config.pca_dim)).cuda()
            self.sanity_val_mean = torch.mean(batch[0]['ppg'], dim=0)
        loss, pred_bp, t_abp, label, group  = self._shared_step(batch, mode='val')
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        self.step_mode = 'test'
        loss, pred_bp, t_abp, label, group = self._shared_step(batch, mode='test')  
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group}  

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        group = torch.cat([v["group"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach(), group)
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
        group_mse = self.grouping(prev_mse, group)
        group_mae = self.grouping(prev_mae, group)
        group_me = self.grouping(prev_me, group)
        return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me} 
    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.prompt_learner_glo.parameters(), 'lr': self.config.lr, 'weight_decay': self.config.wd},])
        return optimizer
