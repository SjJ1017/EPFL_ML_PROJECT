#!/bin/bash

PARAMS=(
    "6 512 30 2"
    "4 512 30 2"
    "6 256 30 2"
    "6 512 30 1"
    "6 512 20 2"
)

for PARAM in "${PARAMS[@]}"; do

    read NUM_LAYERS HIDDEN_DIM EPOCHS GAMMA <<< "$PARAM"

    echo "================ Running experiment ================="
    echo "num_layers=$NUM_LAYERS, hidden_dim=$HIDDEN_DIM, epochs=$EPOCHS, gamma=$GAMMA"

    python experiments_e2e.py \
        --num_layers $NUM_LAYERS \
        --hidden_dim $HIDDEN_DIM \
        --epochs $EPOCHS \
        --gamma $GAMMA
done