#!/bin/bash

echo "Training LSTM Model with Improved Settings"
echo "==========================================="

python train.py \
    --model_type lstm \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0.0002 \
    --optimizer rmsprop \
    --scheduler plateau \
    --bidirectional \
    --min_lr 1e-6 \
    --gradient_clip 1.5 \
    --max_length 256 \
    --vocab_size 10000 \

echo "LSTM training completed!"