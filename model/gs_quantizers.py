"""
Gumbel-Softmax code selection for the BRIDLE tokenizer (GS-BRIDLE / GS-BRIDLE-VQ).

Implements Sections 3.2-3.3 of the manuscript. Equation numbers refer to the manuscript.

  Eq. (11)  logit        l_{m,k} = r_m . c_{m,k} / (||r_m|| ||c_{m,k}||)       (cosine similarity)
  Eq. (12)  probability  p_{m,k} = softmax_k((l_{m,k} + g_{m,k}) / tau),   g ~ Gumbel(0, 1)
  Eq. (13)  selection    y_m = argmax_k p_{m,k}
  Eq. (14)  quantization q_m = c_{m, y_m}                                 (hard code in the forward pass)
  Eq. (15)  output       q = sum_m q_m
            backward     straight-through Gumbel-Softmax: the forward value is the hard code,
                         the gradient flows through p_{m,k} (no identity-based STE).
  Eq. (16)  lambda       lambda = (1/S) log(tau0 / tau_min)
  Eq. (17)  temperature  tau(s) = tau0 * (tau_min / tau0) ** (s / S),  s = training step,
                         S = total tokenizer-training steps (30 epochs), tau0 = 2.0, tau_min = 0.5

Unchanged from the official BRIDLE implementation (the manuscript follows the code):
  * the residual entering each stage is L2-normalized and the next residual is r_m - q_m;
  * codebooks are updated by EMA (decay 0.99), here driven by the selected index y_m;
  * the codebook loss is computed per stage (Eq. (10) with the stage residual) and averaged
    over the M stages, beta = 0.25;
  * target codes for encoder training and all evaluations use the original nearest-code
    quantizer (eval mode / no gradient), so targets are deterministic.

Usage
    tokenizer = BRIDLE_tokenizer.__dict__[args.model](...)          # unchanged
    enable_gumbel(tokenizer.quantizer, tau0=2.0, tau_min=0.5, schedule='exp')
    set_gumbel_total_steps(tokenizer.quantizer, epochs * len(train_loader))   # tokenizer phase
"""
import math
import torch
import torch.nn.functional as F

from .rq_quantizations import VQEmbedding, RQBottleneck
from .quantizations import NormEMAVectorQuantizer, ema_inplace, norm_ema_inplace


def _l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def _gumbel_noise(like):
    u = torch.rand_like(like)
    return -torch.log(-torch.log(u + 1e-10) + 1e-10)


class _TauMixin:
    """Step-wise temperature schedule. The step counter is a buffer, so it is saved in
    checkpoints and resumes correctly; it is reset when a new tokenizer phase starts."""

    def _gs_init(self, tau0, tau_min, schedule):
        assert schedule in ('exp', 'linear', 'fixed'), schedule
        self.gs_tau0, self.gs_tau_min, self.gs_schedule = float(tau0), float(tau_min), schedule
        self.gs_total_steps = None
        if not hasattr(self, 'gumbel_step'):
            self.register_buffer('gumbel_step', torch.zeros((), dtype=torch.long))

    def current_tau(self):
        if self.gs_schedule == 'fixed' or not self.gs_total_steps:
            return self.gs_tau0
        frac = min(float(self.gumbel_step.item()) / self.gs_total_steps, 1.0)
        if self.gs_schedule == 'exp':
            return self.gs_tau0 * (self.gs_tau_min / self.gs_tau0) ** frac
        return self.gs_tau0 + (self.gs_tau_min - self.gs_tau0) * frac      # linear

    def _gs_active(self):
        return self.training and torch.is_grad_enabled()


class VQEmbeddingGS(_TauMixin, VQEmbedding):
    """One RVQ stage of BRIDLE with Gumbel-Softmax selection (used inside RQBottleneck)."""

    def forward(self, input):
        if not self._gs_active():
            return super().forward(input)                      # original BRIDLE: nearest code

        input = _l2norm(input)                                 # same as original
        if self.kmeans_init:
            self.init_embed_(input.reshape(-1, self.embed_dim))
            self.kmeans_init = False

        shape = input.shape
        x = input.reshape(-1, self.embed_dim)
        W = self.weight[:-1, :].detach().clone()               # EMA updates weight in place below
        logits = x @ _l2norm(W).t()                            # cosine similarity
        probs = F.softmax((logits + _gumbel_noise(logits)) / self.current_tau(), dim=-1)
        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, W.shape[0]).type_as(probs)
        quant = ((hard - probs).detach() + probs) @ W          # forward = c_y, backward through probs

        if self.ema:
            self._ema_update_embedding(x, idx)                 # original EMA rule, selected index
        if self.restart_unused_codes:
            self._restart_unused_codes(x, idx)                 # original behaviour (DDP only)

        self.gumbel_step += 1
        return quant.view(shape), idx.view(shape[:-1])


class _RQBottleneckGSMixin:
    """RQBottleneck whose output keeps the Gumbel gradient path instead of the identity STE."""

    def forward(self, z):
        if not (self.training and torch.is_grad_enabled()):
            return RQBottleneck.forward(self, z)               # original (identity STE, unused)
        bsz, seq_len, hidden_dim = z.shape
        z_flat = z.reshape(-1, hidden_dim)
        quant_list, codes, loss_list = self.quantize(z_flat)   # original residual loop and losses
        loss = torch.mean(torch.stack(loss_list))
        z_q = quant_list[-1].view(bsz, seq_len, hidden_dim)    # sum of hard codes, Gumbel gradient
        return z_q, loss, codes.view(bsz, seq_len, -1)


class RQBottleneckGS(_RQBottleneckGSMixin, RQBottleneck):
    pass


class NormEMAVectorQuantizerGS(_TauMixin, NormEMAVectorQuantizer):
    """Single-codebook (VQ) BRIDLE quantizer with Gumbel-Softmax selection."""

    def forward(self, z):
        if not self._gs_active():
            return super().forward(z)                          # original: nearest code + STE

        z = _l2norm(z)
        zf = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(zf)

        W = self.embedding.weight.detach().clone()
        logits = zf @ _l2norm(W).t()
        probs = F.softmax((logits + _gumbel_noise(logits)) / self.current_tau(), dim=-1)
        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, W.shape[0]).type_as(probs)
        z_q = ((hard - probs).detach() + probs) @ W

        if self.embedding.ema:                                 # original EMA rule, selected index
            with torch.no_grad():
                bins = hard.sum(0); self.all_reduce_fn(bins)
                ema_inplace(self.cluster_size, bins, self.decay)
                zero_mask = bins == 0
                bins = bins.masked_fill(zero_mask, 1.)
                embed_sum = zf.detach().t() @ hard; self.all_reduce_fn(embed_sum)
                embed_normalized = _l2norm((embed_sum / bins.unsqueeze(0)).t())
                embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight, embed_normalized)
                norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        loss = F.mse_loss(z_q, zf.detach()) + self.commitment_loss_weight * F.mse_loss(z_q.detach(), zf)
        self.gumbel_step += 1
        return z_q.view(z.shape), loss, idx.view(*z.shape[:-1])


# ── helpers ───────────────────────────────────────────────────────────────
def enable_gumbel(quantizer, tau0=2.0, tau_min=0.5, schedule='exp'):
    """Convert an already-built BRIDLE quantizer (RQBottleneck or NormEMAVectorQuantizer)."""
    if isinstance(quantizer, RQBottleneck):
        quantizer.__class__ = RQBottleneckGS
        for cb in quantizer.codebooks:
            cb.__class__ = VQEmbeddingGS
            cb._gs_init(tau0, tau_min, schedule)
    elif isinstance(quantizer, NormEMAVectorQuantizer):
        quantizer.__class__ = NormEMAVectorQuantizerGS
        quantizer._gs_init(tau0, tau_min, schedule)
    else:
        raise TypeError(f'unsupported quantizer {type(quantizer).__name__}')
    print(f'🔥 Gumbel-Softmax enabled: {type(quantizer).__name__} | tau {tau0} → {tau_min} ({schedule}, step-wise)')
    return quantizer


def _gs_modules(quantizer):
    return list(quantizer.codebooks) if isinstance(quantizer, RQBottleneck) else [quantizer]


def set_gumbel_total_steps(quantizer, total_steps):
    for m in _gs_modules(quantizer):
        m.gs_total_steps = int(total_steps)


def gumbel_tau(quantizer):
    m = _gs_modules(quantizer)[0]
    return m.current_tau(), int(m.gumbel_step.item())
