# L2P method to  Predict BP with PPG signal

## Evrionment and  Dataset

Please refer to this [repository](https://github.com/inventec-ai-center/bp-benchmark) for methods of acquiring and refining datasets

install requirements of ours

## Explanation of the command-line arguments
- `--config_file`: Path to the configuration file for the experiment.
- `--lr`: Learning rate for the optimizer.
- `--wd`: Weight decay (L2 regularization) for the optimizer.
- `--global_coeff`: Coefficient for the global prompt in the loss function.
- `--method` : The training method to use, with options "original" and "prompt_global".
- `--root_dir` : Root directory for the experiment.
- `--result_dirname`: Directory name for storing the experiment results.
- `--glonorm`: normalize prompt before merging it with the input.
- `--normalize`: Normalize prompt+input before it gets into model.
- `--clip`: Use clipping durning prompt+input.
- `--seed`: Seed value for reproducibility.
- `--group_avg`: Use BP group average .
- `--penalty`: Add entropy term in loss term.
- `--score_ratio`: Adjust the ratio of key-query score in the loss term.
- `--k`: How many keys to select.
- `--num_pool`: Number of key & prompt in prompt pool.
- `--lamb`: Lambda value, Adjusts the ratio of entropy in the loss term.
- `--cnn` : apply CNN

## Model and Data Directory

- Model code example
    - code\train\core\load_model.py

    ```
    import os
    root = '/path/to/model'
    join = os.path.join
    model_fold = {"bcg": {0: {0: join(root, 'bcg-resnet1d-fold0-original-seed0.ckpt'),
                        1: join(root, 'bcg-resnet1d-fold1-original-seed0.ckpt'),
                        2: join(root, 'bcg-resnet1d-fold2-original-seed0.ckpt'),
                        3: join(root, 'bcg-resnet1d-fold3-original-seed0.ckpt'),
                        4: join(root, 'bcg-resnet1d-fold4-original-seed0.ckpt')}}}

    ```
- Data

    ```
    ├── L2p
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
## Docker Environment 

```
docker load < /mlainas/yewon/bp_yewon.tar
```

