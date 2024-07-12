#%%
import os
import wandb
import joblib
from shutil import rmtree, copy
import pandas as pd
import numpy as np 
from scipy.io import loadmat
from glob import glob
import pytorch_lightning as pl
from sklearn.decomposition import PCA

# Load loaders
from core.loaders import *
from core.solver_s2s import Solver

#####################################################
#####################################################
from core.utils import (get_nested_fold_idx, get_ckpt, cal_metric, cal_statistics, mat2df, to_group,
                        remove_outlier, group_annot, group_count, group_shot, transferring, extract_pca_statistics, filter_dataset_based_on_statistics)
from core.load_model import model_fold
from core.model_config import model_configuration
import pickle
from omegaconf import OmegaConf
#####################################################
#####################################################
# Load model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from core.models import *
from core.prompt_tuning import Custom_model
# Others
import torch.nn as nn
import torch
import mlflow as mf
import coloredlogs, logging
from pathlib import Path
import warnings
import pdb
warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

# def seed_everything(SEED):
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     pl.utilities.seed.seed_everything(seed=SEED)
#     torch.backends.cudnn.deterministic = True ###
#     torch.backends.cudnn.benchmark = False ###

#%%

    
class SolverS2l(Solver):
    # def __init__(self, config, transfer):
    #     super(SolverS2l, self).__init__()
    #     self.transfer
    def _get_model(self, ckpt_path_abs=None, fold=None):
        model = None
        if self.config.transfer:
            if self.config.exp.model_type == "resnet1d":
                if self.config.transfer == "uci2":
                    fold=0
                backbone_name = f"{self.config.transfer}-{self.config.exp.model_type}"
                if self.config.method == "original": # TODO
                    if self.config.scratch:
                        self.transfer_config_path = f"./core/config/dl/resnet/resnet_{self.config.transfer}.yaml"
                        self.transfer_config = OmegaConf.load(self.transfer_config_path)
                        self.transfer_config = transferring(self.config, self.transfer_config)
                        model = Resnet1d_original(self.transfer_config.param_model, random_state=self.transfer_config.exp.random_state)
                    else:
                        model = Resnet1d_original.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                        lr = self.config.param_model.lr
                        wd = self.config.param_model.wd
                        model.param_model.lr = lr
                        model.param_model.wd = wd
                else:
                    model = Resnet1d.load_from_checkpoint(f"pretrained_models/{backbone_name}/fold{fold}.ckpt")
                # Initialize Classifier
                model.model.main_clf = nn.Linear(in_features=model.model.main_clf.in_features,
                                                 out_features=model.model.main_clf.out_features,
                                                 bias=model.model.main_clf.bias is not None)
                print(f"####### Load {self.config.exp.model_type} backbone model pre-trained by {self.config.transfer} #######")
            else:
                NotImplementedError
            return model

        elif not ckpt_path_abs:
            if self.config.exp.model_type == "resnet1d":
                if self.config.method == "original":
                    model = Resnet1d_original(self.config.param_model, random_state=self.config.exp.random_state)
                else:
                    model = Resnet1d(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet(self.config.param_model, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP(self.config.param_model, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else:
            if self.config.exp.model_type == "resnet1d":
                if self.config.method == "original":
                    model = Resnet1d_original.load_from_checkpoint(ckpt_path_abs)
                else:
                    model = Resnet1d.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet.load_from_checkpoint(ckpt_path_abs)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP.load_from_checkpoint(ckpt_path_abs)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val"):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm

        #--- Predict
        pred = outputs["pred_bp"].numpy()
        true = outputs["true_bp"].numpy()
        naive =  np.mean(dm.train_dataloader(is_print=False).dataset._target_data, axis=0)
        #####################################################
        #####################################################

        # Make Prediction into Group Prediction
        pred_group_bp, true_group_bp = to_group(pred, true, self.config, loader.dataset.bp_norm)  # TODO
        #####################################################
        #####################################################
        #--- Evaluate
        err_dict = {}
        for i, tar in enumerate(['SP', 'DP']):
            tar_acrny = 'sbp' if tar=='SP' else 'dbp'
            pred_bp = bp_denorm(pred[:,i], self.config, tar)
            true_bp = bp_denorm(true[:,i], self.config, tar)
            naive_bp = bp_denorm(naive[i], self.config, tar)

            # error
            err_dict[tar_acrny] = pred_bp - true_bp
            fold_errors[f"{mode}_{tar_acrny}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar_acrny}_label"].append(true_bp)
            #####################################################
            #####################################################

            # Group Performance
            for group in ["hypo", "normal", "prehyper", "hyper2",]:
                if len(pred_group_bp[tar][group]) != 0:
                    pr_group_bp = bp_denorm(pred_group_bp[tar][group], self.config, tar)
                    tr_group_bp = bp_denorm(true_group_bp[tar][group], self.config, tar)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_pred"].append(pr_group_bp)
                    fold_errors[f"{mode}_{tar_acrny}_{group}_label"].append(tr_group_bp)
            #####################################################
            #####################################################
            fold_errors[f"{mode}_{tar_acrny}_naive"].append([naive_bp]*len(pred_bp))
        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        
        metrics = cal_metric(err_dict, mode=mode)    
        return metrics
    
    def get_file_name(self, group_idx):
        class_scale = self.config.class_scale
        emb_dim = 128 # args로 주고 싶다면 1. train.py에 args추가하고 bash파일에 --emb_dim 추가.
        # 그러면 다른 변수처럼 self.config.emb_dim으로 불러올 수 있음.
        aug_type = self.config.aug_type
        aug_shot = self.config.shots
        file_name = f'sample_free_{group_idx}_cs_{class_scale}_tr_{emb_dim}_cnd_{aug_type}_few_{aug_shot}_last.pkl'
        return file_name
    
    def get_setting_name(self):
        emb_dim = 128
        layers = 8
        difftimes = 200
        
        setting_name = f'dim_{emb_dim}_layers_{layers}_difftimes_{difftimes}'
        return setting_name
    
    def smoothing(self, x, alpha=0.1):
        """
        Exponential Weighted Moving Average (EWMA) for a 2D signal with batches.
        
        Parameters:
        - x: A 2D numpy array of shape [batch_size, signal_length].
        - alpha: Smoothing factor within [0, 1], controlling the decay rate.
        
        Returns:
        - A 2D numpy array containing the smoothed signal for each batch.
        """
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0]  # 각 배치의 첫 번째 값을 초기 값으로 설정
        for t in range(1, x.shape[1]):  # signal_length에 대해 반복
            y[:, t] = alpha * x[:, t] + (1 - alpha) * y[:, t-1]
        return y
        
    def slice_tensors(self, seq_tensor, yss, sample_nums):
        # 각 텐서에서 sample_nums에 따라 텐서를 자르는 함수
        sliced_tensors = []
        sliced_ys = []
        for tensor, ys, num in zip(seq_tensor, yss, sample_nums):
            sliced_tensors.append(tensor[:num, :])
            sliced_ys.append(ys[:num, :])
        return sliced_tensors, sliced_ys
    

    def evaluate(self):
        #####################################################
        #####################################################

        # Make Template considering group sbp and group dbp
        fold_errors_template = {"subject_id":[], "record_id": [],
                                "sbp_naive":[],  "sbp_pred":[], "sbp_label":[],
                                "dbp_naive":[],  "dbp_pred":[],   "dbp_label":[],
                                "sbp_hypo_pred":[], "dbp_hypo_pred":[],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        
        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]

        #####################################################
        #####################################################
        # Remove outlier
        all_split_df = remove_outlier(all_split_df)
        
        # Make group annotation
        all_split_df = group_annot(all_split_df) 

        if self.config.count_group:
            group_count(all_split_df)
        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df) 
        print(self.config)

        run_list = []
        logger_list = []

        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            # seed_everything(self.config.seed)
            # get_nested_fold_idx : [[0,1,2],[3],[4]] ## Generator
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            #if foldIdx==1: break

            data_root = self.config.data_root
            data_root = os.path.join(data_root, f'seed_{self.config.seed}')
            setting_name = self.get_setting_name()
            data_root = os.path.join(data_root, setting_name)
            fold_root = os.path.join(data_root, f'fold_{foldIdx}', self.config.exp.data_name)
            trainset_pf = os.path.join(fold_root, 'trainset.pt')
            trainset_dict = torch.load(trainset_pf)
            
            trainset = trainset_dict['ppg'].squeeze(1)
            mask = ~torch.isnan(trainset).any(dim=1)
            trainset = trainset[mask]
            trainset = trainset.detach().cpu().numpy()
            
            ys = trainset_dict['spdp']
            sbp = ys[:, 0]
            dbp = ys[:, 1]

            train_df = pd.DataFrame()
            train_df['SP'] = sbp
            train_df['DP'] = dbp
            train_df['signal'] = [trainset[index_value] for index_value in range(len(trainset))]
            train_df = group_annot([train_df])
            # train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            if self.config.shots: # train and validate with few-shot
                # train_df = group_shot(train_df, n=self.config.shots)
                val_df = group_shot(val_df, n=5)
            
            test_df = pd.concat(np.array(all_split_df)[folds_test]) # test with full data of a fold
            # dm.setup_kfold(train_df, val_df, test_df)
            dm.setup_kfold(train_df[0], val_df, test_df)      
            if self.config.augment_diffusion:
                samples_list = []
                ys_list = []
                class_list = ['hypo', 'normal', 'prehyper', 'hyper2']
                train_num_list = []
                for class_idx, class_name in enumerate(class_list):
                    file_name = self.get_file_name(class_idx)
                    class_pf = os.path.join(fold_root, file_name)

                    # signal_root = self.config.signal_root
                    # signal_root = f'{signal_root}' + self.config['guidance_scale']
                    # signal_path = os.path.join(fold_root, signal_root)
                    # class_pf = os.path.join(signal_path, f'sample_group_average_loss_{class_idx}.pkl')
                    # 일단 100개 가져오는 걸로 짜볼텐데, 나중에 데이터 개수 adaptive하게 가져와야 하면 이 코드 쓰면 됨.
                    # num_sample = self.config['data_distribution'][f'fold_{foldIdx}'][class_name] 
                    num_sample = self.config.aug_shots
                    train_num_list.append(num_sample)
                    
                    with open(class_pf, 'rb') as f:
                        sample_f = pickle.load(f) # 여기가 cpu
                        # sample_f = torch.load(f, map_location=torch.device('cpu:0'))
                        samples = sample_f['sampled_seq'].squeeze(1)
                        mask = ~torch.isnan(samples).any(dim=1)
                        samples = samples[mask]
                        ys = sample_f['y']
                        ys = ys[mask]
                        
                    
 
                    ## filtering
                    
                    if self.config.filtering:
                        gen_group = torch.full((samples.shape[0],), class_idx)
                        boundary = self.config.filtering_boundary
                        n_components=10
                        pca = pca = PCA(n_components=n_components)
                        train_pca_statistics = extract_pca_statistics(data=samples,
                                                                    labels=None,
                                                                    groups=gen_group,
                                                                    pca=pca,
                                                                    boundary=boundary)

                        
                        filtered_data, ys = filter_dataset_based_on_statistics(data=samples, 
                                                                        labels=ys, 
                                                                        groups=gen_group,
                                                                        statistics=train_pca_statistics,
                                                                        pca=pca,
                                
                                                                        is_pca=True)
                        
                        samples = filtered_data[class_idx]
                        
                    samples_list.append(samples)
                    ys_list.append(ys)     
                sample_nums = train_num_list
                samples, ys = self.slice_tensors(samples_list, ys_list, sample_nums)
                
                samples = torch.vstack(samples).detach().cpu().numpy()
                if self.config.smoothing:
                    samples = self.smoothing(samples)
                ys = torch.vstack(ys).detach().cpu().numpy()
             
                sbp = ys[:, 0]
                dbp = ys[:, 1]
                # DataFrame 생성 및 컬럼 추가
                df = pd.DataFrame()
                df['SP'] = sbp
                df['DP'] = dbp
                df['signal'] = [samples[index_value] for index_value in range(len(samples))]
                dm.folds_train = pd.concat([dm.folds_train, df]) 
            dm.folds_train['trial'] = '3047369_0006_0'
            dm.folds_train['patient'] = 543

            dm.folds_train = dm.folds_train.reset_index(drop=True)
            if 'abp_signal' in dm.folds_train.columns:
                dm.folds_train.drop('abp_signal', axis=1, inplace=True)
            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max] 

            #####################################################
            #####################################################
            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                ck_path = os.path.join(self.config.root_dir, "models", model_fold[self.config.backbone][data_name][self.config.seed][foldIdx])
                if self.config.transfer:
                    regressor = self._get_model(fold=foldIdx)
                else:
                    regressor = self._get_model(ck_path) # Load Model # TODO
                model_config = model_configuration[self.config.backbone][data_name] # Load pre-trained model config
                data_shape = model_config["data_dim"] 
                model = Custom_model(regressor, data_shape, model_config, self.config, stats) # Call Prompt Model 

                # Gradient Control -- freeze pre-train model // train prompt
                for name, param in model.named_parameters():
                    if 'prompt_learner' not in name:
                        param.requires_grad_(False)
                    if self.config.update_encoder:
                        if 'extractor' in name:
                            param.requires_grad_(True)
                    if self.config.update_regressor:
                        if 'regressor' in name:
                            param.requires_grad_(True)
                enabled = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        enabled.add(name)
                print(f"Parameters to be updated: {enabled}")

            if self.config.method == "original":
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:
                    model = self._get_model()
                print("##"*20)
                for name, param in model.named_parameters():
                    
                    # Linear-Probing
                    if self.config.lp: 
                        if not "main_clf" in name:
                            param.requires_grad_(False)
                            send_message = "Linear Probing"

                    # Fine-tuning
                    elif self.config.transfer:
                        send_message = "Fine-tuning"

                    # Scratch
                    else:
                        send_message = "Training from Scratch"
                print(send_message)
                print("##"*20)

                enabled = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        enabled.add(name)
                print(f"Update Param:\n{enabled}")
                print("##"*20)
            #####################################################
            #####################################################
            # Define optimizer with different learning rates for prompt_learner and other parameters

            early_stop_callback = EarlyStopping(**dict(self.config.param_early_stop)) # choosing earlystopping policy
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt)) # choosing model save policy
            lr_logger = LearningRateMonitor()

            #import pdb; pdb.set_trace()
            # Control all training and test process
            if not self.config.shots:
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])
            else: 
                trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback, lr_logger ])

            #--- trainer main loop
            logger_list.append(trainer.logger.log_dir)
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # train
                trainer.fit(model, dm) ## Training
                print("run_id", run.info.run_id)

                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id)) 
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0]) # mlruns ckpt_path
                print(ckpt_path_abs)
                if self.config.shots:
                    ckpt_path_abs = checkpoint_callback.best_model_path
                print("Best model path: {}".format(ckpt_path_abs))

                # ### Sanity Check: Model Weight Update
                # if self.config.sanity_check:
                #     print(f"encoder_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                #     print(f"regressor_update: {model.state_dict()['regressor.model.first_block_conv.conv.weight'][0][0][0]}")
                
                # load best ckpt
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"]) # TODO

                # evaluate
                val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False) ## Validation
                test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True) ## Training

                if self.config.gen_ip: # if gen_ip=True, Save prompt data, ppg_input, global_prompt, instance-wise prompt, merged prompt
                    with open(f"./glogen_glogen_fold_{foldIdx}.pkl", 'wb') as pickle_file:
                        pickle.dump(test_outputs, pickle_file)               
                
                # save updated model
                trainer.model = model
                trainer.save_checkpoint(ckpt_path_abs)

                metrics = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

                
                redundant_model_path = str(Path(artifact_uri))
                run_list.append(redundant_model_path)
                

            #--- Save to model directory
            if self.config.save_checkpoint:
                os.makedirs("{}-{}/{}".format(self.config.path.model_directory, self.config.method, self.config.exp.exp_name), exist_ok=True)
                ckpt_real = "{}-{}/{}/fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(
                                                                            self.config.path.model_directory,
                                                                            self.config.method,
                                                                            self.config.exp.exp_name,
                                                                            foldIdx,
                                                                            metrics["test/sbp_mae"],
                                                                            metrics["test/dbp_mae"])
                trainer.save_checkpoint(ckpt_real)
                print(ckpt_real)

        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)

        #####################################################
        #####################################################
        for mode in ['val', 'test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp',  'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            #####################################################
            #####################################################
            tmp_metric = cal_metric(err_dict, mode=mode)
            if mode == 'test':
                sbp = tmp_metric['test/sbp_mae']
                dbp = tmp_metric['test/dbp_mae']
                groups = ['hypo', 'normal', 'prehyper', 'hyper2']
                sbps = [tmp_metric[f'test/sbp_{group}_mae'] for group in groups]
                dbps = [tmp_metric[f'test/dbp_{group}_mae'] for group in groups]
                sbp_gal = sum(sbps) / 4
                dbp_gal = sum(dbps) / 4
                gal = sbp_gal + dbp_gal
                tmp_metric['sbp_gal'] = sbp_gal
                tmp_metric['dbp_gal'] = dbp_gal
                tmp_metric['gal'] = gal
                
                if not self.config.ignore_wandb:
                    wandb.log(tmp_metric)
                    wandb.run.summary['sbp_gal'] = sbp_gal
                    wandb.run.summary['dbp_gal'] = dbp_gal
                    wandb.run.summary['gal'] = gal
                    wandb.run.summary['sbp'] = sbp
                    wandb.run.summary['dbp'] = dbp
                    wandb.run.summary['spdp'] = sbp + dbp
            out_metric.update(tmp_metric)

        return out_metric, run_list, logger_list

    def test(self): # Similar with self.evaluate() - all same but no training [trainer.fit()]
        results = {}
        #####################################################
        #####################################################

        fold_errors_template = {"subject_id": [], "record_id": [],
                                "sbp_naive": [], "sbp_pred": [], "sbp_label": [],
                                "dbp_naive": [], "dbp_pred": [], "dbp_label": [],
                                "sbp_hypo_pred": [], "dbp_hypo_pred": [],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                }
        #####################################################
        #####################################################
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]}

        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]

        #####################################################
        #####################################################
        all_split_df = remove_outlier(all_split_df)
        all_split_df = group_annot(all_split_df)
        #####################################################
        #####################################################
        #--- Nested cv
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            #seed_everything(self.config.seed)
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  
                break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            dm.setup_kfold(train_df, val_df, test_df)

            # Find scaled ppg_max, ppg_min
            ppg_min = np.min(dm.train_dataloader().dataset.all_ppg)
            ppg_max = np.max(dm.train_dataloader().dataset.all_ppg)
            stats = [ppg_min, ppg_max] 

            #--- load trained model
            if 'param_trainer' in self.config.keys():
                trainer = MyTrainer(**dict(self.config.param_trainer))
            else:
                trainer = MyTrainer()

            ckpt_path_abs = glob(f'{self.config.param_test.model_path}{foldIdx}' + '*.ckpt')[0]

            if self.config.method.startswith("prompt"):
                #--- Init model\ ##
                data_name = self.config.exp["data_name"]
                if self.config.transfer:
                    regressor = self._get_model(fold=foldIdx)
                else:
                    regressor = self._get_model(ckpt_path_abs) # Load Model # TODO
                model_config = model_configuration[self.config.backbone][data_name] # Load pre-trained model config
                data_shape = model_config["data_dim"] 
                model = Custom_model(regressor, data_shape, model_config, self.config, stats)
                #model = Custom_model.load_from_checkpoint(ckpt_path_abs) # Call Prompt Model 
                #import pdb; pdb.set_trace()
                model.load_state_dict(torch.load(ckpt_path_abs)["state_dict"])

                #model = model.load_from_checkpoint(ckpt_path_abs)

            if self.config.method == "original":
                if self.config.transfer:
                    model = self._get_model(fold=foldIdx)
                else:
                    model = self._get_model()

            model.eval()
            trainer.model = model

            #--- get test output
            val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
            test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
            
            metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
            logger.info(f"\t {metrics}")

        #--- compute final metric
        results['fold_errors'] = fold_errors
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        #####################################################
        #####################################################
        for mode in ['test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'dbp', 'sbp_hypo', 'dbp_hypo', 'sbp_normal', 'dbp_normal',
                                    'sbp_prehyper', 'dbp_prehyper', 'sbp_hyper2', 'dbp_hyper2',
                                    ]}
            tmp_metric = cal_metric(err_dict, mode=mode)

            out_metric.update(tmp_metric)
        #####################################################
        #####################################################
        results['out_metric'] = out_metric
        os.makedirs(os.path.dirname(self.config.param_test.save_path), exist_ok=True)
        joblib.dump(results, self.config.param_test.save_path)

        print(out_metric)