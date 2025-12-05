#!/bin/bash

for i in {0..60}; do
    start=$((i * 1000))
    end=$((start + 1000))

    echo "Running chunk $i: start_idx=$start"

    # 最后一个不需要 end_idx
    if [ $i -eq 6 ]; then
        python model.py \
            --split test --train_split 0.99 \
            --hidden_dim 512 --num_layers 8 \
            --model_save_path .cache/best_larger_train.pth \
            --epoch 30 --batch_size 32 --split train \
            --data_cache .cache/data_cache_train.pkl \
            --features_cache .cache/features_cache_train.pkl \
            --start_idx $start \
            --feature_split $i
    else
        python model.py \
            --split test --train_split 0.99 \
            --hidden_dim 512 --num_layers 8 \
            --model_save_path .cache/best_larger_train.pth \
            --epoch 30 --batch_size 32 --split train \
            --data_cache .cache/data_cache_train.pkl \
            --features_cache .cache/features_cache_train.pkl \
            --start_idx $start \
            --end_idx $end \
            --feature_split $i
    fi
done