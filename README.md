# GS-BRIDLE: Gumbel-Softmax-Based Residual Vector Quantization for BRIDLE

Code for “Improving Self-Supervised Audio Representation Learning via Gumbel-Softmax-Based Residual Vector Quantization” ([journal, year], [DOI]).

GS-BRIDLE keeps the architecture and training procedure of BRIDLE and replaces only the code selection of its tokenizer: at every RVQ stage, the nearest-code selection with the straight-through estimator (STE) is replaced by the **straight-through Gumbel-Softmax estimator** with exponential temperature annealing.

This repository is built on the official BRIDLE implementation (commit `c83d1ac`; the original README is kept as [README_BRIDLE.md](README_BRIDLE.md)). The original MIT license and copyright notice of BRIDLE are retained in [LICENSE](LICENSE).

## Changes to BRIDLE

| File | Change |
|---|---|
| `model/gs_quantizers.py` | **new**. Gumbel-Softmax versions of the BRIDLE quantizers (RVQ: `VQEmbeddingGS`, `RQBottleneckGS`; single codebook: `NormEMAVectorQuantizerGS`) |
| `main_pretrain.py` | arguments `--use_gumbel --gumbel_tau --gumbel_min_tau --gumbel_schedule`, `enable_gumbel(...)` right after the tokenizer is built, total number of steps for the temperature schedule |
| `tests/check_gs.py`, `scripts/` | **new**. Implementation check and the pretraining / evaluation commands used in the paper |

In addition, the following compatibility fixes for single-GPU training with recent PyTorch, NumPy and torchaudio were applied to the BRIDLE code and used in all experiments; they do not change the method:

| File | Fix |
|---|---|
| all scripts | `torch.load(..., weights_only=False)` (PyTorch ≥ 2.6) |
| `model/rq_quantizations.py` | `dist.all_reduce` is skipped when `torch.distributed` is not initialized |
| `dataset.py` | audio is read with `soundfile` instead of `torchaudio.load` |
| `main_finetune.py`, `unit_finetune.py` | `global_rank` initialized; `args.criterion` replaced by `args.multilabel` |
| `util/pos_embed.py`, `util_video/kinetics.py` | NumPy 2 compatibility (`np.float` → `float`, unused import removed) |
| `model_video/vit.py` | **new**. Small re-export module required by `model_video/encoder_decoder.py` |

All other files (models, losses, data pipeline, fine-tuning) are those of BRIDLE.

## Correspondence with the manuscript

| Manuscript | Implementation |
|---|---|
| Eq. (11) logit = cosine similarity between the stage residual and each code vector | `logits = x @ normalize(W).t()`, `x` = L2-normalized residual |
| Eq. (12) Gumbel-Softmax probability | `probs = softmax((logits + g) / tau)`, `g ~ Gumbel(0, 1)` |
| Eq. (13) selected code index | `idx = probs.argmax(-1)` |
| Eq. (14) hard quantization in the forward pass | `quant = ((onehot(idx) - probs).detach() + probs) @ W` — the value is exactly `c_{m,y}`; the gradient flows through `probs` |
| Eq. (15) sum over the M stages | `RQBottleneckGS.forward` returns the sum of the stage outputs **without** the identity STE `z + sg[q - z]` of BRIDLE |
| Eq. (16)–(17) exponential annealing, tau0 = 2.0, tau_min = 0.5, step-wise | `current_tau()`; S = (tokenizer epochs) × (iterations per epoch); the step counter is stored in the checkpoint |
| EMA codebook update (Sec. 3.1) | original BRIDLE EMA rule, driven by the selected index `idx` |
| Target generation for the main encoder | Gumbel noise is **not** used: in eval mode / without gradients, the original BRIDLE quantizer (nearest code) is used, so targets are deterministic |

### Difference from the STE of BRIDLE

| | BRIDLE (STE) | GS-BRIDLE (straight-through Gumbel-Softmax) |
|---|---|---|
| Selection | `y = argmin_k ||r - c_k||` | `y = argmax_k (l_k + g_k)` |
| Forward | `c_y` | `c_y` |
| Backward | `dq/dr ≈ I` (identity) | `dq/dl = W^T dp/dl`, `p = softmax((l + g)/tau)` |
| Code | `z_q = z + sg[q - z]` | `q = (onehot(y) - p).detach() @ W + p @ W` |

## Conventions inherited from the BRIDLE code

| Item | Implementation |
|---|---|
| Encoder loss (Eq. (8)) | cross-entropy over masked positions only, averaged over masked positions and codebooks |
| Cosine loss (Eq. (9)) | per-position cosine similarity, averaged over positions |
| Tokenizer loss | `L_codebook + L_cos` (lambda_cos = 1) |
| Codebook loss (Eq. (10)) | per stage between the normalized stage residual and the stage code, averaged over stages, beta = 0.25 |
| Residual | each stage input is L2-normalized; next residual = normalized residual − selected code |

## Usage

```bash
pip install timm einops soundfile librosa tensorboard torchaudio torchvision decord

# 1) check that the implementation matches the manuscript
python tests/check_gs.py
#   [RQ] OK | encoder grad ... | tau [2.0, 1.0, 0.5]
#   [VQ] OK | encoder grad ... | tau [2.0, 1.0, 0.5]

# 2) pretraining on AS-20K (main encoder 130 ep -> tokenizer 30 ep -> main encoder 130 ep)
bash scripts/pretrain.sh rq            # GS-BRIDLE    (M = 4, K = 256)
bash scripts/pretrain.sh vq            # GS-BRIDLE-VQ (M = 1, K = 1,024)
GUMBEL="" bash scripts/pretrain.sh rq  # BRIDLE baseline with identical settings

# 3) AS-20K fine-tuning and linear probing
bash scripts/evaluate_as20k.sh outputs/rq/iter2_encoder/checkpoint-129.pth
```

During tokenizer training, the log prints the temperature at the start of each epoch (`Gumbel tau at epoch start: 2.0000 ... ≈ 0.52`).

## Downstream evaluation settings

| Dataset | Setting source | Max. epochs | Learning rate |
|---|---|---|---|
| AS-20K (FT) | BRIDLE | 150 | 4e-4 |
| AS-20K (LP) | BRIDLE | 400 | 4e-2 |
| ESC-50 (FT, LP) | BRIDLE (FT) | 300 | 2e-3 |
| UrbanSound8K (FT, LP) | ARCH (LP) | 200 | 1e-3 |
| VocalSound (FT, LP) | – | 50 | 1e-4 |

AdamW, weight decay 1e-4, batch size 32 (single GPU), early stopping within the maximum number of epochs with the same criterion for all models. 2D time-frequency masking (ratio 0.2) is used only for fine-tuning on AS-20K and ESC-50.

## Pretrained checkpoints

| Model | Link |
|---|---|
| GS-BRIDLE (AS-20K) | [link] |
| GS-BRIDLE-VQ (AS-20K) | [link] |

## Citation

```
[BibTeX]
```

Tested on a single GPU (no distributed training).
