#!/bin/bash

echo "Training GPT Model with Improved Settings"
echo "=========================================="

python train.py \
    --model_type gpt \
    --hidden_dim 768 \
    --num_layers 12 \
    --dropout 0.5 \
    --use_pretrained \
    --epochs 8 \
    --batch_size 16 \
    --learning_rate 3e-6 \
    --weight_decay 0.02 \
    --optimizer adamw \
    --scheduler cosine \
    --min_lr 1e-7 \
    --gradient_clip 1.5 \
    --max_length 256 \
    --early_stopping \
    --early_stopping_patience 3

echo "GPT training completed!"