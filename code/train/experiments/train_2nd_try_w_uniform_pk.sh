#!/bin/bash
cd ..
GPU_IDS=(2 3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0

TRAINING_SCRIPT="train.py"

# Fixed Hyper-para
GLONORM_OPTIONS=("")
PENALTY_SCALE_RANGE=(0)
WEIGHT_PER_PROMPT_OPTIONS=("")
SCALING_OPTIONS=('--normalize')
POOL_RANGE=(10)
QK_SIM_COEFF_RANGE=(0)
PCA_DIM_RANGE=(20)

# Search range
LR_RANGE=(1e-2 1e-3 1e-4)
WD_RANGE=(1e-1 1e-2 1e-3)
PROMPT_WEIGHTS_OPTIONS=('learnable' 'attention')
BATCHSIZE_RANGE=(20 4)
QUERY_DIM_RANGE=(32 128)
HEAD_OPTIONS=("" '--train_head' '--train_head --reset_head')
GLOBAL_COEFF_RANGE=(3 0.3)