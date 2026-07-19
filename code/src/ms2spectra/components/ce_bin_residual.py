import torch as th
import torch.nn as nn

from ms2spectra.utils.misc_utils import scatter_logsumexp


class CEBinResidualHead(nn.Module):
    """
    CE-conditioned final-bin residual head.

    It operates after R98 binned spectrum rendering.
    The last layer is zero-initialized, so before training it is exactly no-op.
    """

    def __init__(self, input_size=14, hidden_size=128, dropout=0.05):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)


def _get_batch_size(batch_size):
    if hasattr(batch_size, "detach"):
        return int(batch_size.detach().cpu().item())
    return int(batch_size)


def _get_ce_per_spec(batch, batch_size, device, dtype):
    ce = batch.get("spec_ce", None)
    bs = _get_batch_size(batch_size)

    if ce is None:
        return th.zeros(bs, device=device, dtype=dtype)

    ce = ce.to(device=device, dtype=dtype).reshape(-1)

    if ce.numel() == bs:
        return ce

    ce_batch_idxs = batch.get("spec_ce_batch_idxs", None)
    if ce_batch_idxs is None:
        if ce.numel() == 1:
            return ce.expand(bs)
        return th.zeros(bs, device=device, dtype=dtype)

    ce_batch_idxs = ce_batch_idxs.to(device=device).long().reshape(-1).clamp(0, bs - 1)

    ce_sum = th.zeros(bs, device=device, dtype=dtype)
    ce_cnt = th.zeros(bs, device=device, dtype=dtype)

    ce_sum.index_add_(0, ce_batch_idxs, ce)
    ce_cnt.index_add_(0, ce_batch_idxs, th.ones_like(ce))

    return ce_sum / ce_cnt.clamp_min(1.0)


def apply_ce_bin_residual(
    head,
    hparams,
    batch,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
    true_prec_mzs,
    batch_size,
):
    """
    Apply CE-conditioned residual to final binned prediction.

    Returns:
        pred_mzs, new_logprobs, pred_batch_idxs, stat
    """

    if head is None:
        return pred_mzs, pred_logprobs, pred_batch_idxs, None

    if pred_mzs.numel() == 0:
        return pred_mzs, pred_logprobs, pred_batch_idxs, None

    bs = _get_batch_size(batch_size)
    device = pred_mzs.device
    dtype = pred_logprobs.dtype

    batch_idx = pred_batch_idxs.long().clamp(0, bs - 1)

    mz_max = max(float(getattr(hparams, "mz_max", 1500.0)), 1.0)
    ce_max = max(float(getattr(hparams, "ce_max", 100.0)), 1.0)

    prec = true_prec_mzs.to(device=device, dtype=dtype).reshape(-1)
    if prec.numel() < bs:
        pad = th.full((bs - prec.numel(),), mz_max, device=device, dtype=dtype)
        prec = th.cat([prec, pad], dim=0)
    prec = prec[:bs]
    prec_i = prec[batch_idx].clamp_min(1.0)

    ce_per_spec = _get_ce_per_spec(
        batch=batch,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    ce_i = ce_per_spec[batch_idx]

    mz = pred_mzs.to(dtype=dtype)
    nloss = (prec_i - mz).clamp(min=0.0)

    mz_norm = (mz / mz_max).clamp(0.0, 1.5)
    prec_norm = (prec_i / mz_max).clamp(0.0, 1.5)
    nloss_norm = (nloss / mz_max).clamp(0.0, 1.5)
    rel_mz = (mz / prec_i).clamp(0.0, 2.0)

    lp_norm = (pred_logprobs.clamp(min=-30.0, max=0.0) / 30.0).to(dtype=dtype)
    prob = pred_logprobs.exp().clamp(min=0.0, max=1.0).to(dtype=dtype)
    sqrt_prob = prob.sqrt()

    ce_norm = (ce_i / ce_max).clamp(0.0, 2.0)
    ce_norm2 = ce_norm * ce_norm
    ce20 = th.sigmoid((ce_i - 20.0) / 10.0)
    ce40 = th.sigmoid((ce_i - 40.0) / 10.0)

    is_low_mz = (mz < 200.0).to(dtype=dtype)
    is_mid_mz = ((mz >= 200.0) & (mz < 600.0)).to(dtype=dtype)
    is_high_mz = (mz >= 600.0).to(dtype=dtype)
    is_prec_near = ((prec_i - mz).abs() < 2.0).to(dtype=dtype)

    feat = th.stack(
        [
            mz_norm,
            prec_norm,
            nloss_norm,
            rel_mz,
            lp_norm,
            prob,
            sqrt_prob,
            ce_norm,
            ce_norm2,
            ce20,
            ce40,
            is_low_mz,
            is_mid_mz,
            is_high_mz + is_prec_near,
        ],
        dim=1,
    )

    feat = th.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

    raw_delta = head(feat)

    scale = float(getattr(hparams, "ce_bin_residual_delta_scale", 0.10))
    delta = scale * th.tanh(raw_delta)

    old_lse = scatter_logsumexp(
        pred_logprobs,
        pred_batch_idxs,
        dim_size=bs,
    )

    new_logits = pred_logprobs + delta

    new_lse = scatter_logsumexp(
        new_logits,
        pred_batch_idxs,
        dim_size=bs,
    )

    new_logprobs = new_logits - new_lse[batch_idx] + old_lse[batch_idx]

    stat = {
        "delta_abs_mean": delta.detach().abs().mean(),
        "delta_mean": delta.detach().mean(),
    }

    return pred_mzs, new_logprobs, pred_batch_idxs, stat
