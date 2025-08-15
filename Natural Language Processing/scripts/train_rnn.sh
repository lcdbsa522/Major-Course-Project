# #!/bin/bash

# echo "Training RNN Model"
# echo "=================="

# python train.py \
#     --model_type rnn \
#     --embedding_dim 128 \
#     --hidden_dim 256 \
#     --num_layers 2 \
#     --dropout 0.3 \
#     --epochs 20 \
#     --batch_size 32 \
#     --learning_rate 0.001 \
#     --max_length 256 \
#     --vocab_size 10000 \
#     --save_dir 'results/rnn_1'

# echo "RNN training completed!"

#!/bin/bash

echo "Training RNN Model with Improved Settings"
echo "=========================================="

python train.py \
    --model_type rnn \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.1 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --weight_decay 0.0001 \
    --optimizer sgd \
    --scheduler plateau \
    --gradient_clip 1.0 \
    --max_length 256 \
    --vocab_size 10000 \

echo "RNN training completed!"