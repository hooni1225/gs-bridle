#!/usr/bin/env bash
# GS-BRIDLE / GS-BRIDLE-VQ pretraining on AudioSet-20K (manuscript Sec. 4.2).
#   bash scripts/pretrain.sh rq    # GS-BRIDLE    (M = 4, K = 256)
#   bash scripts/pretrain.sh vq    # GS-BRIDLE-VQ (M = 1, K = 1024)
#   GUMBEL="" bash scripts/pretrain.sh rq   # BRIDLE baseline with identical settings
set -e
KIND=${1:-rq}
DATA_TRAIN=${DATA_TRAIN:-data/audioset/bal_train.json}
LABEL_CSV=${LABEL_CSV:-data/audioset/class_labels_indices.csv}
OUT=${OUT:-outputs/${KIND}}

if [ "$KIND" = "rq" ]; then
  CB="--codebook_type rq --codebook_set 4 --code_num 256 --code_dim 256"
else
  CB="--codebook_type legacy --codebook_set 1 --code_num 1024 --code_dim 256"
fi
CB="$CB --ema --kmeans_init --commitment_loss_weight 0.25"                 # beta = 0.25
GUMBEL=${GUMBEL---use_gumbel --gumbel_tau 2.0 --gumbel_min_tau 0.5 --gumbel_schedule exp}   # Eq. (16)-(17)
COMMON="--dataset audioset --data_train $DATA_TRAIN --label_csv $LABEL_CSV --model vit_b \
        --weight_decay 1e-4 --batch_size 16 --precision fp32 --seed 42 --save_every_epoch 5"

# Phase 1: iteration-1 main encoder (cold-start tokenizer), 130 epochs, lr 5e-4, mask 0.8
python main_pretrain.py $COMMON $CB --train_encoder --cold_start --epochs 130 --lr 5e-4 --mask_ratio 0.8 \
       --output_dir $OUT/iter1_encoder --log_dir $OUT/iter1_encoder

# Phase 2: tokenizer, 30 epochs, lr 2e-4, no masking, Gumbel-Softmax with tau 2.0 -> 0.5
python main_pretrain.py $COMMON $CB $GUMBEL --epochs 30 --lr 2e-4 \
       --prev_phase $OUT/iter1_encoder/checkpoint-129.pth \
       --output_dir $OUT/iter1_tokenizer --log_dir $OUT/iter1_tokenizer

# Phase 3: iteration-2 main encoder, 130 epochs, lr 5e-4, mask 0.8 (targets: nearest codes of the frozen tokenizer)
python main_pretrain.py $COMMON $CB $GUMBEL --train_encoder --epochs 130 --lr 5e-4 --mask_ratio 0.8 \
       --prev_phase $OUT/iter1_tokenizer/checkpoint-29.pth \
       --output_dir $OUT/iter2_encoder --log_dir $OUT/iter2_encoder
