# GS-BRIDLE

Code for “Improving Self-Supervised Audio Representation Learning via Gumbel-Softmax-Based Residual Vector Quantization” ([journal, year], [DOI]).

GS-BRIDLE keeps the architecture and training procedure of [BRIDLE](https://arxiv.org/abs/2502.02118) and changes only how its tokenizer selects codes: at every RVQ stage, the nearest-code selection with the identity straight-through estimator (STE) is replaced by the **straight-through Gumbel-Softmax estimator** with exponential temperature annealing (τ: 2.0 → 0.5).

This repository is built on the official BRIDLE implementation (commit `c83d1ac`, MIT License; original README in [README_BRIDLE.md](README_BRIDLE.md)).

## What is changed

| File | Change |
|---|---|
| `model/gs_quantizers.py` | **new** — Gumbel-Softmax versions of the BRIDLE quantizers (RVQ and single codebook) |
| `main_pretrain.py` | options `--use_gumbel --gumbel_tau --gumbel_min_tau --gumbel_schedule` |
| `tests/check_gs.py` | **new** — checks that the forward output equals the selected code, that gradients reach the tokenizer encoder, and the temperature schedule |
| `scripts/` | **new** — pretraining and AS-20K evaluation commands used in the paper |

Minor compatibility fixes for single-GPU training with recent PyTorch, NumPy and torchaudio were also applied (e.g., `torch.load(..., weights_only=False)`, skipping `dist.all_reduce` without distributed training); they do not change the method.

Gumbel noise is used only while training the tokenizer. When the trained tokenizer produces targets for the main encoder, codes are selected deterministically as in BRIDLE.

## Usage

```bash
pip install timm einops soundfile librosa tensorboard torchaudio torchvision decord

# check the implementation
python tests/check_gs.py
#   [RQ] OK | encoder grad ... | tau [2.0, 1.0, 0.5]
#   [VQ] OK | encoder grad ... | tau [2.0, 1.0, 0.5]

# pretraining on AS-20K (main encoder 130 ep -> tokenizer 30 ep -> main encoder 130 ep)
bash scripts/pretrain.sh rq            # GS-BRIDLE    (M = 4, K = 256)
bash scripts/pretrain.sh vq            # GS-BRIDLE-VQ (M = 1, K = 1,024)
GUMBEL="" bash scripts/pretrain.sh rq  # BRIDLE baseline with identical settings

# AS-20K fine-tuning and linear probing
bash scripts/evaluate_as20k.sh outputs/rq/iter2_encoder/checkpoint-129.pth
```

Downstream evaluation settings are given in the paper.

