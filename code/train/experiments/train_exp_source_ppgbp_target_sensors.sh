#!/bin/bash
cd ..
GPU_IDS=(0 1 2 3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0

TRAINING_SCRIPT="train.py"

# Define the fixed parameters
TRANSFER="ppgbp"
TARGET="sensors"
CONFIG_FILE="core/config/dl/resnet/resnet_${TARGET}.yaml"
METHOD="prompt_global"
BACKBONE="resnet1d"
SHOTS=5
PENALTY=""

# Fixed Hyper-para
GLONORM_OPTIONS=("")
PENALTY_SCALE_RANGE=(0)
WEIGHT_PER_PROMPT_OPTIONS=("")
SCALING_OPTIONS=('--normalize')
QK_SIM_COEFF_RANGE=(0)
PCA_DIM_RANGE=(20)

# Search range
POOL_RANGE=(10) #4 10 20)
LR_RANGE=(1e-1 1e-2) # 1e-3) # 1e-4)
WD_RANGE=(1e-1 1e-2) # 1e-3) #(1e-1 
PROMPT_WEIGHTS_OPTIONS=('attention')
BATCHSIZE_RANGE=(4)
QUERY_DIM_RANGE=(256)
HEAD_OPTIONS=("") # '--train_head' '--train_head --reset_head')
GLOBAL_COEFF_RANGE=(5)

for LR in "${LR_RANGE[@]}"
do
for WD in "${WD_RANGE[@]}"
do
for POOL in "${POOL_RANGE[@]}"
do
for PS in "${PENALTY_SCALE_RANGE[@]}"
do
for GLONORM in "${GLONORM_OPTIONS[@]}"
do
for GC in "${GLOBAL_COEFF_RANGE[@]}"
do
for QK in "${QK_SIM_COEFF_RANGE[@]}"
do
for PCADIM in "${PCA_DIM_RANGE[@]}"
do
for WPP in "${WEIGHT_PER_PROMPT_OPTIONS[@]}"
do
for SO in "${SCALING_OPTIONS[@]}"
do
for PW in "${PROMPT_WEIGHTS_OPTIONS[@]}"
do
for BZ in "${BATCHSIZE_RANGE[@]}"
do
for HD in "${HEAD_OPTIONS[@]}"
do
for QD in "${QUERY_DIM_RANGE[@]}"
do

CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
--config_file $CONFIG_FILE \
--method $METHOD \
--backbone $BACKBONE \
--shots $SHOTS \
--transfer $TRANSFER \
--target $TARGET \
$SO \
$GLONORM \
$PENALTY \
$WPP \
$HD \
--query_dim $QD \
--lr $LR \
--batch_size $BZ \
--wd $WD \
--num_pool $POOL \
--global_coeff $GC \
--qk_sim_coeff $QK \
--pca_dim $PCADIM \
--prompt_weights $PW \
--penalty_scaler $PS &

IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))
if [ $IDX -eq 0 ]; then
wait
fi
done
done
done
done
done
done
done
done
done
done
done
done
done
done