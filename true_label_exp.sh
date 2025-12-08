#!/bin/bash

# 可选：训练样本和验证样本
TRAIN_SAMPLES=${1:-None}
VAL_NUM=${2:-2000}

# 参数组合列表：num_layers hidden_dim epochs gamma
PARAMS=(
    "6 512 30 2"
    "4 512 30 2"
    "6 256 30 2"
    "6 512 30 1"
    "6 512 20 2"
)

for PARAM in "${PARAMS[@]}"; do
    # 拆分参数
    read NUM_LAYERS HIDDEN_DIM EPOCHS GAMMA <<< "$PARAM"

    echo "================ Running experiment ================="
    echo "num_layers=$NUM_LAYERS, hidden_dim=$HIDDEN_DIM, epochs=$EPOCHS, gamma=$GAMMA"

    python experiments_e2e.py \
        --num_layers $NUM_LAYERS \
        --hidden_dim $HIDDEN_DIM \
        --epochs $EPOCHS \
        --gamma $GAMMA \
        --train_samples $TRAIN_SAMPLES \
        --val_num $VAL_NUM
done