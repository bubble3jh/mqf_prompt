# Multi-Query Frequency Prompting for Physiological Signal Domain Adaptation

This is the official pytorch implementation of the paper "Multi-Query Frequency Prompting for Physiological Signal Domain Adaptation, Kang & Byun et al.".

## Environment and  Dataset

Please refer to this [repository](https://github.com/inventec-ai-center/bp-benchmark) for methods of acquiring and refining datasets


## Explanation of the command-line arguments
- `--transfer`: Source dataset used to train the pre-trained model.
- `--target`: Target dataset for domain adaptation.
- `--config_file`: Path to the configuration file for the experiment, which is according to the target dataset.
- `--lr`: Learning rate for the optimizer.
- `--wd`: Weight decay (L2 regularization) for the optimizer.
- `--global_coeff`: Coefficient for the prompt, adding original signal.
- `--method` : "original" for the baselines and "prompt_global" for ours.
- `--normalize`: Normalize prompted input to distribution of source dataset.
- `--clip`: Clipping to the original signal's range.
- `--num_pool`: Number of key & prompt in prompt pool.
- `--train_imag`: Whether to train the imag. region.
- `--train_head`: Whether to train the last head layer of the pre-trained model.
- `--reset_head`: Whether to reset the weights of the last head layer of the pre-trained model.
- `--trunc_dim`: Size of prompting area of signal's frequency.
- `--query_dim`: Size of query and key.
- `--pca_dim`: Size of reduction dimension of PCA.
- `--shots`: Trainable data to handle a few-shot ratio.
- `--use_emb_diff`: Use auxiliary loss to make emb closer to souce emb.
- `--diff_loss_weight`: Scaler of auxiliary loss.

## Model and Data Directory

- Model code example
    - code\train\core\load_model.py

    ```
    import os
    root = '/path/to/model'
    join = os.path.join
    model_fold = {"bcg": {0: {0: join(root, 'bcg-resnet1d/fold0.ckpt'),
                        {1: join(root, 'bcg-resnet1d/fold1.ckpt'),
                        {2: join(root, 'bcg-resnet1d/fold2.ckpt'),
                        {3: join(root, 'bcg-resnet1d/fold3.ckpt'),
                        {4: join(root, 'bcg-resnet1d/fold4.ckpt')}}}

    ```
- Data

    ```
    ├── bp-algorithm
    ├── datasets
    │   ├── splits
    │   │   ├── bcg_dataset
    │   │   │   ├── feat_fold_0.csv
    │   │   │   ├── feat_fold_1.csv
    │   │   │   ├── feat_fold_2.csv
    │   │   │   ├── feat_fold_3.csv
    │   │   │   ├── feat_fold_4.csv

    ```


## Implement Code Example

```
pip intall -r requirements.txt

cd L2P_code/code/train

#BCG (10epoch)
python train.py python train.py --lamb 1 --config_file core/config/dl/resnet/resnet_bcg.yaml --method prompt_global --k 1 --num_pool 20  --lr 1e-2 --penalty --glonorm --cnn

#PPGBP (100epoch)
python train.py python train.py --lamb 1 --config_file core/config/dl/resnet/resnet_ppgbp.yaml --method prompt_global --k 1 --num_pool 10  --lr 1e-3 --wd 1e-3 --cnn

#Sensors (5epcoh)
python train.py python train.py --lamb 1 --config_file core/config/dl/resnet/resnet_sensors.yaml --method prompt_global --k 1 --num_pool 10  --lr 1e-2 --wd 1e-3 --cnn

```

## Dataset & model
```
cp /mlainas/yewon/bp-benchmark/bp-algorithm/datasets bp-algorithm/datasets
cp /mlainas/yewon/bp-benchmark/bp-algorithm/code/train/real_model bp-algorithm/code/train/real_model

```


