#!/bin/bash

echo "Training BERT Model with Improved Settings"
echo "==========================================="

python train.py \
    --model_type bert \
    --hidden_dim 768 \
    --dropout 0.5 \
    --use_pretrained \
    --epochs 6 \
    --batch_size 16 \
    --learning_rate 2e-6 \
    --weight_decay 0.02 \
    --optimizer adamw \
    --scheduler plateau \
    --scheduler_patience 2 \
    --scheduler_factor 0.5 \
    --min_lr 1e-7 \
    --gradient_clip 1.5 \
    --max_length 256 \
    --early_stopping \
    --early_stopping_patience 5

echo "BERT training completed!"