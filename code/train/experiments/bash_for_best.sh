#!/bin/bash
cd ..
GPU_IDS=(0 1 2 3 4 5 6 7)  # 사용할 GPU ID 리스트
IDX=0

TRAINING_SCRIPT="train.py"

# SCALING_OPTIONS=('' '--clip' '--normalize')
# HEAD_OPTIONS=('--train_head' '--train_head' '--reset_head')
# PCA_DIM_RANGE=(8 16 32)
PASS_PCA_OPTIONS=("")
IMAG_ON=("--train_imag")
TRUNC_DIM=(50 25)

for PP in "${PASS_PCA_OPTIONS[@]}"
do
for IG in "${IMAG_ON[@]}"
do
for TD in "${TRUNC_DIM[@]}"
do

CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --config_file core/config/dl/resnet/resnet_bcg.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer ppgbp --target bcg --query_dim 16 --lr 1e-4 --batch_size 4 --wd 1e-2 --num_pool 3 --global_coeff 20 --qk_sim_coeff 0 --pca_dim 16 --lam 1.0 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --config_file core/config/dl/resnet/resnet_bcg.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer sensors --target bcg --query_dim 64 --lr 1e-3 --batch_size 4 --wd 1e-3 --num_pool 10 --global_coeff 20 --qk_sim_coeff 0 --pca_dim 16 --lam 1.0 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --config_file core/config/dl/resnet/resnet_bcg.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer sensors --target bcg --query_dim 64 --lr 1e-4 --batch_size 4 --wd 1e-2 --num_pool 3 --global_coeff 20 --qk_sim_coeff 0 --pca_dim 16 --lam 1.0 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --clip --train_head --config_file core/config/dl/resnet/resnet_ppgbp.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer bcg --target ppgbp --query_dim 64 --lr 1e-1 --batch_size 10 --wd 1e-3 --num_pool 3 --global_coeff 10 --qk_sim_coeff 0 --pca_dim 16 --lam 1 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq --pass_pca
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --clip --train_head --config_file core/config/dl/resnet/resnet_ppgbp.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer sensors --target ppgbp --query_dim 16 --lr 1e-3 --batch_size 20 --wd 1e-2 --num_pool 10 --global_coeff 20 --qk_sim_coeff 0 --pca_dim 16 --lam 1.0 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq
CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python train.py $PP $IG --trunc_dim $TD --clip --train_head --config_file core/config/dl/resnet/resnet_sensors.yaml --method prompt_global --backbone resnet1d --shots 5 --transfer ppgbp --target sensors --query_dim 64 --lr 1e-2 --batch_size 4 --wd 1e-1 --num_pool 10 --global_coeff 20 --qk_sim_coeff 0 --pca_dim 16 --lam 1.0 --prompt_weights learnable --penalty_scaler 0 --stepbystep --add_freq

IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))
if [ $IDX -eq 0 ]; then
wait
fi
done
done
done