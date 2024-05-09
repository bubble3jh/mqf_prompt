import os
import joblib
from shutil import rmtree
import pandas as pd
import numpy as np
from scipy.io import loadmat
from glob import glob

# Load loaders
from core.loaders import *
from core.solver_s2s import Solver
from core.utils import (get_nested_fold_idx, get_ckpt, cal_metric, cal_statistics, mat2df)

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
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str, help="model")
    parser.add_arugment("--backbone", type=str)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
