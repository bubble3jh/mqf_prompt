import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as pl
# Load modules
from core.solver_s2s import Solver as solver_s2s
from core.solver_s2l import SolverS2l as solver_s2l
from core.solver_f2l import SolverF2l as solver_f2l
from core.utils import log_params_mlflow, init_mlflow, save_result

from omegaconf import OmegaConf
from time import time, ctime
import mlflow as mf
from shutil import rmtree
from pathlib import Path

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

def seed_everything(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    pl.utilities.seed.seed_everything(seed=SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--global_coeff", type=float, default=1.0)
    parser.add_argument("--gen_coeff", type=float, default=1.0)
    parser.add_argument("--method", choices=["original", "prompt_global"])
    parser.add_argument("--root_dir", default="./")
    parser.add_argument("--result_dirname", default="results")
    parser.add_argument("--layer_num", default=0, type=int)
    parser.add_argument("--glonorm", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_avg", action='store_true')
    parser.add_argument("--penalty", action='store_true')
    parser.add_argument("--k" , type=int, default=1)
    parser.add_argument("--num_pool" , type=int, default=10)
    parser.add_argument("--mul" , action='store_true')
    parser.add_argument("--fixed_key" , action='store_true')
    parser.add_argument("--fixed_prompt" , action='store_true')  
    parser.add_argument("--trans" , action='store_true')  
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
    # -------------------------------------------------------
    parser.add_argument("--use_group" , action='store_true') # set default to FALSE  
    parser.add_argument("--backbone", choices=["resnet1d", "mlpbp", "spectroresnet"], required=True)
    parser.add_argument("--shots", default=0, type=int, help="Few-shot Regression")
    parser.add_argument("--transfer", default=None, type=str, choices=["ppgbp", "sensors", "uci2", "bcg"])
    parser.add_argument("--target", default=None, type=str, choices=["ppgbp", "sensors", "uci2", "bcg"])
    parser.add_argument("--prompt_weights", default='learnable', type=str, choices=["learnable", "cos_sim"])
    parser.add_argument("--penalty_scaler" , type=float, default=0.1)
    parser.add_argument("--qk_sim_coeff", type=float, default=0.5)
    parser.add_argument("--pca_dim", default=20, type=int)
    parser.add_argument("--weight_per_prompt", action='store_true', help="on=> (3, pool), off => (3) learable weight")
    
    parser.add_argument("--lp", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    return parser

def parser_to_config(parser, config):
    """
    Add parser argument into oemgaconfig
    """
    args = vars(parser.parse_args())
    for key, item in args.items():
        config[key] = item
        if key == "lr" or key == "wd":
            config.param_model[key] = item
    return config

def main(args):
    # Config File Exist?
    if os.path.exists(args.config_file) == False:
        raise RuntimeError("config_file {} does not exist".format(args.config_file))

    time_start = time()
    config = OmegaConf.load(args.config_file)
    result_dir = os.path.join(args.root_dir, f"{args.result_dirname}/{config.exp['model_type']}/{config.exp['data_name']}")
    os.makedirs(result_dir, exist_ok = True)
    if args.mul:
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}_mul.csv")
    else: 
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}.csv")

    if args.penalty:
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}_penalty.csv")
    if args.fixed_key:
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}_fixkey.csv")
    if args.fixed_prompt:
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}_noise.csv")
    if args.trans:
        result_name = os.path.join(result_dir, f"{args.method}_top_{args.k}_seed{args.seed}_trans.csv")
    
    config = parser_to_config(parser, config) # To preserve config file but input the argument

    if config.group_avg:
        val_type = "val_group_mse"
        config.objective.type = val_type
        config.param_early_stop.monitor = val_type
        config.logger.param_ckpt.monitor = val_type

    # import pdb; pdb.set_trace()
    # set seed
    seed_everything(config.seed)

    config.param_trainer.max_epochs=100
    config.param_trainer.check_val_every_n_epoch=2
    config.param_model.batch_size=256
    config.param_early_stop.patience=10
    config.exp.N_fold=5

    #--- get the solver
    if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
        solver = solver_s2s(config)
    elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp']: # Our Interest
        torch.use_deterministic_algorithms(True)
        solver = solver_s2l(config)
    else:
        solver = solver_f2l(config)

    #--- training and logging into mlflow
    init_mlflow(config) # TODO: Didn't consider mlflow, in experiment for etri, we use csv file in the path of result_name
    with mf.start_run(run_name=f"{config.exp.N_fold}fold_CV_Results") as run:
        log_params_mlflow(config)
        cv_metrics = solver.evaluate() # TODO Training and Evaluation
        logger.info(cv_metrics)
        mf.log_metrics(cv_metrics)
    #"original", "prompt_global", "prompt_gen", "prompt_glogen", "prompt_lowgen", "prompt_lowglogen"
        # cv_metrics['name'] : The name of each run for sweep
        if config.method == 'original':
            cv_metrics['name'] = f"original_lr_{config.lr}_wd_{config.wd}"
        elif config.method == 'prompt_lowgen' or config.method == 'prompt_lowglogen':
            cv_metrics['name'] = f'layer_{config.layer_num}_lr_{config.lr}_wd_{config.wd}'
            
        elif config.method == 'prompt_global':
            cv_metrics['name'] = f"lr_{config.lr}_wd_{config.wd}"
            cv_metrics['name'] += f'_k_{config.k}'
            cv_metrics['name'] += f'_lambda_{config.qk_sim_coeff}'
            cv_metrics['name'] += f'_num_pool_{config.num_pool}'
            if config.penalty:
                cv_metrics['name'] += f'_penalty'
            
        elif config.method == 'prompt_gen' or config.method == 'prompt_glogen':
            cv_metrics['name'] = f'lr_{config.lr}_wd_{config.wd}'
        if config.method in ['prompt_gen', 'prompt_glogen', 'prompt_lowglogen', 'prompt_lowgen']:
            cv_metrics['name'] += f'_gencoef_{config.gen_coeff}'
        if config.method in ['prompt_global', 'prompt_glogen', 'prompt_lowglogen' ]:
            cv_metrics['name'] += f'_global_coeff_{config.global_coeff}'

        if config.glonorm:
            cv_metrics['name'] += '_glonorm'
        if config.group_avg:
            cv_metrics['name'] += '_group_avg'
        if config.normalize:
            cv_metrics['name'] += '_norm'
        if config.clip:
            cv_metrics['name'] += '_clip'
        if config.seed != 0:
            cv_metrics['name'] += f'_seed_{config.seed}'
        if config.num_pool != 10:
            cv_metrics['name'] += f'_numprompt_{config.seed}'
        

        # save_result(cv_metrics, result_name) # Save test result to csv
    time_now = time()
    logger.warning(f"Time Used: {ctime(time_now-time_start)}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    src_data = args.transfer
    tgt_data = args.target
    group_name = f'{src_data}-{tgt_data}'
    if args.scratch:
        group_name = group_name + '-scratch'
    if args.lp:
        group_name = group_name + '-linearprobing'
    if args.method == 'prompt_global':
        group_name = group_name + f'-prompt_global'
    
    if not args.ignore_wandb:
        import wandb
        wandb.init(entity='l2p_bp', project='fewshot_transfer', group=group_name)
        lr = args.lr
        wd = args.wd
        run_name = f'seed:{args.seed}-lr:{lr}-wd:{wd}'
        wandb.run.name = run_name
        wandb.config.update(args)
        
    main(parser.parse_args())

    wandb.finish()