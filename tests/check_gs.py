"""
Sanity check that the Gumbel-Softmax quantizers behave as described in the manuscript.
Run from the repository root (GPU recommended):  python tests/check_gs.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import model.tokenizer as TK
from model.gs_quantizers import enable_gumbel, set_gumbel_total_steps, gumbel_tau


def build(kind):
    kw = dict(in_chans=1, audio_exp=True, img_size=(1024, 128), mode=0, pos_trainable=False, estimator_mode=2,
              code_dim=256, commitment_loss_weight=0.25, ema=True, kmeans_init=True, soft_code=1, no_shift=False)
    if kind == 'rq':
        kw.update(codebook_type='rq', code_num=256, codebook_set=4)
    else:
        kw.update(codebook_type='legacy', code_num=1024, codebook_set=1)
    return TK.vit_b(**kw)


def stages(q):
    return list(q.codebooks) if hasattr(q, 'codebooks') else [q]


def weights(m):
    return (m.weight[:-1] if hasattr(m, 'n_embed') else m.embedding.weight).detach().clone()


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    x = torch.randn(2, 1, 1024, 128, device=dev)
    for kind in ['rq', 'vq']:
        tok = build(kind).to(dev)
        enable_gumbel(tok.quantizer, tau0=2.0, tau_min=0.5, schedule='exp')
        set_gumbel_total_steps(tok.quantizer, 100)

        # 1) gradient reaches the tokenizer encoder through the Gumbel-Softmax probabilities
        tok.train()
        emb, loss = tok.forward_estimator_embeddings(x)
        (emb.pow(2).mean() + loss).backward()
        g = sum(p.grad.norm() ** 2 for n, p in tok.named_parameters()
                if n.startswith('blocks') and p.grad is not None) ** 0.5
        assert g > 0, 'no gradient reached the tokenizer encoder'

        # 2) forward value equals the selected code vector (Eq. (14)): straight-through estimator
        m0 = stages(tok.quantizer)[0]
        z = torch.nn.functional.normalize(torch.randn(16, 256, device=dev), dim=-1)
        W = weights(m0)
        if hasattr(m0, 'n_embed'):
            q, idx = m0(z)
        else:
            q, _, idx = m0(z.view(1, 16, 256)); q, idx = q.view(16, 256), idx.view(16)
        assert torch.allclose(q.detach(), W[idx], atol=1e-5), 'forward value is not the hard code vector'

        # 3) target generation (eval, no gradient) is deterministic
        tok.eval()
        with torch.no_grad():
            assert torch.equal(tok(x), tok(x)), 'targets are not deterministic'

        # 4) temperature schedule of Eq. (17): 2.0 -> 1.0 -> 0.5
        taus = []
        for s in (0, 50, 100):
            for m in stages(tok.quantizer):
                m.gumbel_step.fill_(s)
            taus.append(round(gumbel_tau(tok.quantizer)[0], 4))
        assert taus == [2.0, 1.0, 0.5], taus
        print(f'[{kind.upper()}] OK  | encoder grad {g.item():.3e} | tau {taus}')


if __name__ == '__main__':
    main()
