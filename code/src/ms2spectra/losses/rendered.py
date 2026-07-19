import torch as th

from ms2spectra.losses.base import (
    sparse_cosine_distance,
    sparse_jensen_shannon_divergence,
)


def _r180_ce_vector(batch, batch_size, device, dtype):
    ce = batch.get("spec_ce", None)
    if ce is None:
        raise RuntimeError("R180 requires spec_ce in batch")

    bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, "detach") else int(batch_size)

    ce = ce.reshape(-1).to(device=device, dtype=dtype)

    if int(ce.numel()) == bs:
        return ce

    ce_batch_idxs = batch.get("spec_ce_batch_idxs", None)
    if ce_batch_idxs is None:
        if int(ce.numel()) == 1:
            return ce.expand(bs)
        raise RuntimeError(f"R180 cannot map spec_ce shape={tuple(ce.shape)} to batch_size={bs}")

    ce_batch_idxs = ce_batch_idxs.reshape(-1).to(device=device).long()
    if int(ce_batch_idxs.numel()) != int(ce.numel()):
        raise RuntimeError(
            f"R180 spec_ce/spec_ce_batch_idxs mismatch: {tuple(ce.shape)} vs {tuple(ce_batch_idxs.shape)}"
        )

    ce_sum = th.zeros(bs, device=device, dtype=dtype)
    ce_cnt = th.zeros(bs, device=device, dtype=dtype)

    ce_sum.scatter_add_(0, ce_batch_idxs.clamp(0, bs - 1), ce)
    ce_cnt.scatter_add_(0, ce_batch_idxs.clamp(0, bs - 1), th.ones_like(ce))

    ce_vec = ce_sum / ce_cnt.clamp_min(1.0)
    ce_vec = th.nan_to_num(ce_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return ce_vec


def _r180_ce_weight(ce_vec, hparams):
    mid_thr = float(getattr(hparams, "r180_mid_ce_threshold", 20.0))
    high_thr = float(getattr(hparams, "r180_high_ce_threshold", 40.0))

    low_w = float(getattr(hparams, "r180_low_ce_weight", 0.5))
    mid_w = float(getattr(hparams, "r180_mid_ce_weight", 2.5))
    high_w = float(getattr(hparams, "r180_high_ce_weight", 5.0))

    valid = th.isfinite(ce_vec)

    w = th.ones_like(ce_vec)
    w = th.where(valid & (ce_vec < mid_thr), th.ones_like(w) * low_w, w)
    w = th.where(valid & (ce_vec >= mid_thr), th.ones_like(w) * mid_w, w)
    w = th.where(valid & (ce_vec >= high_thr), th.ones_like(w) * high_w, w)

    return w


def r180_ce_weighted_spectrum_loss(
    batch,
    batch_size,
    hparams,
    true_mzs,
    true_logprobs,
    true_batch_idxs,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
):
    """
    R180: direct spectrum-level CE-weighted loss.

    Returns a dict with per-spectrum vectors:
      loss_vec = ce_weight * (cos_weight * sparse_cosine_distance + jss_weight * sparse_jss_distance)

    This is train-only. It changes training objective, not eval target.
    """
    device = pred_logprobs.device
    dtype = pred_logprobs.dtype

    bs = int(batch_size.detach().cpu().item()) if hasattr(batch_size, "detach") else int(batch_size)

    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        z = th.zeros(bs, device=device, dtype=dtype)
        return {
            "loss_vec": z,
            "cos_dist": z,
            "jss_dist": z,
            "ce_weight": th.ones_like(z),
        }

    mz_max = float(getattr(hparams, "r180_mz_max", getattr(hparams, "mz_max", 1500.0)))
    mz_bin_res = float(getattr(hparams, "r180_mz_bin_res", 0.01))
    log_min = float(getattr(hparams, "log_min", 1.0e-9))

    cos_w = float(getattr(hparams, "r180_cos_weight", 1.0))
    jss_w = float(getattr(hparams, "r180_jss_weight", 0.35))

    cos_dist = sparse_cosine_distance(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_idxs,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        pred_batch_idxs=pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        log_distance=False,
    )

    jss_dist = sparse_jensen_shannon_divergence(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_idxs,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        pred_batch_idxs=pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        log_min=log_min,
    )

    cos_dist = th.nan_to_num(cos_dist, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    jss_dist = th.nan_to_num(jss_dist, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    ce_vec = _r180_ce_vector(batch, batch_size, device=device, dtype=dtype)
    ce_weight = _r180_ce_weight(ce_vec, hparams)

    loss_vec = ce_weight * (cos_w * cos_dist + jss_w * jss_dist)
    loss_vec = th.nan_to_num(loss_vec, nan=0.0, posinf=10.0, neginf=0.0)

    return {
        "loss_vec": loss_vec,
        "cos_dist": cos_dist.detach(),
        "jss_dist": jss_dist.detach(),
        "ce_weight": ce_weight.detach(),
    }
