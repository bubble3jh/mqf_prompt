# Multi-Query Frequency Prompting for Physiological Signal Domain Adaptation

This is the official pytorch implementation of the paper "Multi-Query Frequency Prompting for Physiological Signal Domain Adaptation, Kang & Byun et al.".

## Quick Summary

![Motivation_v2](https://github.com/user-attachments/assets/29397798-9e08-43fe-9acd-f0a800080a9a)
Adding a prompt in the frequency domain and reconstructing the prompted signal is much more robust to pulse shift.

![prompting_method_v2](https://github.com/user-attachments/assets/bc81898c-904c-41b4-838a-5d3346d5d7e6)
Selecting prompts with various features of the signal can make synthesizing prompts flexible and interpretable.

![comparison_ppg_mod](https://github.com/user-attachments/assets/974c8365-d165-4e1f-b17e-d8a2a2fb634d)
As a result, frequency prompting is much more plausible than vanilla prompting to signal and achieves better performance at 10-shot transfer learning, with just updating 1.6% of the weight of the full fine-tuning method.
![image](https://github.com/user-attachments/assets/24014140-c3c4-42ef-a86a-4c19eb560d30)

## Abstract
In this study, we address challenges in applying deep learning to photoplethysmography (PPG) signals for blood pressure prediction, including restricted data availability, domain shifts, and inadequate focus on minority groups with hypertension or hypotension. We propose a novel prompt-learning framework that employs frequency-domain prompting to enhance the robustness of pre-trained models against physiological pulse shifts without extensive retraining. By integrating learnable prompts based on critical signal features such as variability, periodicity, and locality, our method enables more precise adaptations to diverse datasets and conditions. Experimental results across three benchmark datasets demonstrate a significant improvement in accuracy, with a 41.67% reduction in mean absolute error compared to baseline models, alongside a reduction in trainable parameters by up to 98.4%. This approach boosts performance and promotes equitable, efficient healthcare diagnostics across diverse settings.

## Environment and  Dataset

Please refer to this [repository](https://github.com/inventec-ai-center/bp-benchmark) for methods of acquiring and refining datasets


## Explanation of the command-line arguments
- `--transfer`: Source dataset used to train the pre-trained model.
- `--target`: Target dataset for domain adaptation.
- `--config_file`: Path to the configuration file for the experiment, which is according to the target dataset.
- `--lr`: Learning rate for the optimizer.
- `--wd`: Weight decay (L2 regularization) for the optimizer.
- `--batch_size`: Batch size.
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
pip install -r requirements.txt

cd /code/train

#Sweep all train setting
bash experiments/bash_for_bash.sh

#Sensors to BCG transfer 
python train.py --transfer=sensors --target=bcg --batch_size=6 --clip=false --config_file=core/config/dl/resnet/resnet_bcg.yaml --diff_loss_weight=0.4 --epochs=10 --global_coeff=0.1 --lr=0.009 --method=prompt_global --normalize=true --num_pool=9 --pca_dim=16 --query_dim=16 --reset_head=false --shots=10 --train_head=true --train_imag=true --trunc_dim=50 --use_emb_diff=true --wd=0.03

```



