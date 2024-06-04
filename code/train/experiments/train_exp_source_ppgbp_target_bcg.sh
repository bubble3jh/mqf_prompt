#!/bin/bash
cd ..
GPU_IDS=(3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0

TRAINING_SCRIPT="train.py"
LOG_DIR="./experiments/logs"
mkdir -p $LOG_DIR

# Define the fixed parameters
TRANSFER="ppgbp"
TARGET="bcg"
CONFIG_FILE="core/config/dl/resnet/resnet_${TARGET}.yaml"
METHOD="prompt_global"
BACKBONE="resnet1d"
SHOTS=5
PENALTY=""

POOL_RANGE=(4 10)
LR_RANGE=(1e-2 1e-3 1e-4)
WD_RANGE=(1e-1 1e-2 1e-3)
PENALTY_SCALE_RANGE=(0)
GLONORM_OPTIONS=("")
WEIGHT_PER_PROMPT_OPTIONS=("")
PROMPT_WEIGHTS_OPTIONS=('learnable' 'cos_sim')
SCALING_OPTIONS=('--clip' '--normalize')
GLOBAL_COEFF_RANGE=(0.3 1)
QK_SIM_COEFF_RANGE=(0)
PCA_DIM_RANGE=(20 4)
BATCHSIZE_RANGE=(20 4)

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
LOG_FILE="$LOG_DIR/training_lr${LR}_wd${WD}_penalty${PS}_QKsim${QK}_pool${POOL}_PCADIM${PCADIM}_glonorm${GLONORM:+on}_WPP${WPP:+on}_PW${PW:+on}_groupavg${GROUP_AVG:+on}_gc${GC}_$(date +'%Y%m%d_%H%M%S').log"

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
--lr $LR \
--batch_size $BZ \
--wd $WD \
--num_pool $POOL \
--global_coeff $GC \
--qk_sim_coeff $QK \
--pca_dim $PCADIM \
--prompt_weights $PW \
--penalty_scaler $PS > $LOG_FILE 2>&1 &

# Check if the script ran successfully
if [ $? -eq 0 ]; then
echo "Training completed successfully with lr=$LR, wd=$WD, penalty=${PENALTY}. Logs can be found at $LOG_FILE"
else
echo "Training failed with lr=$LR, wd=$WD, penalty=${PENALTY}. Check logs for more details: $LOG_FILE"
fi

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