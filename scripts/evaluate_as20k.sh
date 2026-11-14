#!/usr/bin/env bash
# AS-20K fine-tuning (FT) and linear probing (LP) of a pretrained main encoder (manuscript Sec. 4.2, Table Y).
#   bash scripts/evaluate_as20k.sh outputs/rq/iter2_encoder/checkpoint-129.pth
set -e
CKPT=$1
DATA_TRAIN=${DATA_TRAIN:-data/audioset/bal_train.json}
DATA_EVAL=${DATA_EVAL:-data/audioset/eval.json}
LABEL_CSV=${LABEL_CSV:-data/audioset/class_labels_indices.csv}
OUT=${OUT:-outputs/eval}
COMMON="--dataset audioset --data_train $DATA_TRAIN --data_eval $DATA_EVAL --label_csv $LABEL_CSV \
        --model vit_b --nb_classes 527 --finetune $CKPT --batch_size 32 --weight_decay 1e-4"

# FT: BRIDLE setting, at most 150 epochs, 2D time-frequency masking 0.2
python main_finetune.py $COMMON --lr 4e-4 --epochs 150 \
       --mask_t_prob 0.2 --mask_f_prob 0.2 --mask_2d True \
       --output_dir $OUT/ft --log_dir $OUT/ft

# LP: BRIDLE setting, 400 epochs, frozen encoder
python main_finetune.py $COMMON --linear_probe --lr 4e-2 --epochs 400 --precision fp32 \
       --output_dir $OUT/lp --log_dir $OUT/lp
