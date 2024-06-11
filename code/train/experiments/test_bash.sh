#!/bin/bash

# Move one directory up
cd ..

# Define GPU IDs
GPU_IDS=(1 2 3 4)  
IDX=0

# Define the training script
TRAINING_SCRIPT="train.py"

# Fixed parameters
TRANSFER="bcg"
TARGET="ppgbp"
CONFIG_FILE="core/config/dl/resnet/resnet_${TARGET}.yaml"
METHOD="prompt_global"
BACKBONE="resnet1d"
SHOTS=5
PENALTY=""

# Fixed Hyper-parameters
PENALTY_SCALE_RANGE=(0)
WEIGHT_PER_PROMPT_OPTIONS=("")
QK_SIM_COEFF_RANGE=(0)
GLONORM_OPTIONS=("")
SCALING_OPTIONS=("")
HEAD_OPTIONS=("") 
LAMBDA_RANGE=(1.0)
PCA_DIM_RANGE=(16)

# Search ranges
POOL_RANGE=(3 10)
LR_RANGE=(1e-1 1e-2 1e-3)
WD_RANGE=(1e-1 1e-2 1e-3)
GLOBAL_COEFF_RANGE=(10 5 20)
BATCHSIZE_RANGE=(4 10 20)
QUERY_DIM_RANGE=(16 64)

# Method options
METHOD_OPTIONS=("--stepbystep")
ADD_FREQ=("--add_freq")

# Iterate over all parameter combinations
for M in "${METHOD_OPTIONS[@]}"; do
for AF in "${ADD_FREQ[@]}"; do
for LR in "${LR_RANGE[@]}"; do
for WD in "${WD_RANGE[@]}"; do
for POOL in "${POOL_RANGE[@]}"; do
for PS in "${PENALTY_SCALE_RANGE[@]}"; do
for GLONORM in "${GLONORM_OPTIONS[@]}"; do
for GC in "${GLOBAL_COEFF_RANGE[@]}"; do
for QK in "${QK_SIM_COEFF_RANGE[@]}"; do
for PCADIM in "${PCA_DIM_RANGE[@]}"; do
for WPP in "${WEIGHT_PER_PROMPT_OPTIONS[@]}"; do
for SO in "${SCALING_OPTIONS[@]}"; do
for BZ in "${BATCHSIZE_RANGE[@]}"; do
for HD in "${HEAD_OPTIONS[@]}"; do
for QD in "${QUERY_DIM_RANGE[@]}"; do
for LAM in "${LAMBDA_RANGE[@]}"; do

# Debugging output to see current combination
# echo "Running combination: LR=$LR WD=$WD POOL=$POOL PS=$PS GLONORM=$GLONORM GC=$GC QK=$QK PCADIM=$PCADIM WPP=$WPP SO=$SO BZ=$BZ HD=$HD QD=$QD LAM=$LAM GPU=${GPU_IDS[$IDX]}"

# Comment out the actual training command for debugging purposes
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
--config_file $CONFIG_FILE \
--method $METHOD \
--backbone $BACKBONE \
--shots $SHOTS \
--transfer $TRANSFER \
--target $TARGET \
--query_dim $QD \
--lr $LR \
--batch_size $BZ \
--wd $WD \
--num_pool $POOL \
--global_coeff $GC \
--qk_sim_coeff $QK \
--pca_dim $PCADIM \
--lam $LAM \
--prompt_weights $WPP \
--penalty_scaler $PS \
$M \
$AF \
$SO \
$GLONORM \
$PENALTY \
$HD &

# Update IDX and wait if necessary
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
done
done