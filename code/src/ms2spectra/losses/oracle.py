import torch as th

from ms2spectra.utils.spec_utils import batched_bin_func
from ms2spectra.utils.misc_utils import scatter_l1normalize


def setup_oracle_teacher_bins(pl_module):
    pl_module._oracle_teacher_bins = None
    pl_module._oracle_teacher_meta = None

    if not getattr(pl_module.hparams, "use_oracle_teacher_bin_loss", False):
        return

    pt = getattr(pl_module.hparams, "oracle_teacher_bin_pt", None)
    if pt is None or str(pt).strip() == "":
        raise RuntimeError("use_oracle_teacher_bin_loss=True requires oracle_teacher_bin_pt")

    try:
        payload = th.load(pt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = th.load(pt, map_location="cpu")

    if "teacher" not in payload:
        raise RuntimeError(f"oracle teacher file missing 'teacher': {pt}")

    pl_module._oracle_teacher_bins = payload["teacher"]
    pl_module._oracle_teacher_meta = payload.get("meta", {})

    print(
        "[R28 OracleTeacher] enabled: "
        f"pt={pt}, specs={len(pl_module._oracle_teacher_bins)}, "
        f"meta={pl_module._oracle_teacher_meta}, "
        f"weight={getattr(pl_module.hparams, 'oracle_teacher_bin_loss_weight', None)}"
    )


def oracle_teacher_bin_loss(
    pl_module,
    unique_id,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
):
    eps = float(getattr(pl_module.hparams, "oracle_teacher_bin_eps", 1e-9))
    eps = max(eps, 1e-12)

    teacher = getattr(pl_module, "_oracle_teacher_bins", None)
    batch_size = int(unique_id.shape[0])

    if teacher is None:
        z = pred_logprobs.sum() * 0.0
        return (
            z + th.zeros([batch_size], dtype=pred_logprobs.dtype, device=pred_logprobs.device),
            th.zeros([batch_size], dtype=th.bool, device=pred_logprobs.device),
        )

    if pred_mzs.numel() == 0:
        z = pred_logprobs.sum() * 0.0
        return (
            z + th.zeros([batch_size], dtype=pred_logprobs.dtype, device=pred_logprobs.device),
            th.zeros([batch_size], dtype=th.bool, device=pred_logprobs.device),
        )

    mz_bin_res = float(
        getattr(
            pl_module.hparams,
            "oracle_teacher_bin_mz_bin_res",
            pl_module.hparams.mz_bin_res,
        )
    )
    mz_max = float(pl_module.hparams.mz_max)
    agg = "sum" if pl_module.hparams.sum_ints else "amax"

    pred_probs = pred_logprobs.exp()

    pred_bin_idxs, pred_bin_probs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_probs,
        pred_batch_idxs,
        mz_max,
        mz_bin_res,
        agg,
        sparse=True,
    )

    if pred_bin_idxs.numel() == 0:
        z = pred_logprobs.sum() * 0.0
        return (
            z + th.zeros([batch_size], dtype=pred_logprobs.dtype, device=pred_logprobs.device),
            th.zeros([batch_size], dtype=th.bool, device=pred_logprobs.device),
        )

    pred_bin_probs = scatter_l1normalize(
        pred_bin_probs,
        pred_bin_batch_idxs,
    ).clamp_min(eps)

    loss_vec = th.zeros([batch_size], dtype=pred_logprobs.dtype, device=pred_logprobs.device)
    valid_mask = th.zeros([batch_size], dtype=th.bool, device=pred_logprobs.device)

    min_target = float(getattr(pl_module.hparams, "oracle_teacher_bin_min_target_value", 0.0))

    for b in range(batch_size):
        spec_id = int(unique_id[b].detach().cpu().item())
        item = teacher.get(spec_id, None)
        if item is None:
            continue

        target_value = float(item.get("target_value", 1.0))
        if target_value < min_target:
            continue

        t_bins = item["bin_idxs"].to(device=pred_logprobs.device).long()
        t_probs = item["probs"].to(device=pred_logprobs.device, dtype=pred_logprobs.dtype)

        if t_bins.numel() == 0 or t_probs.numel() == 0:
            continue

        t_probs = t_probs / t_probs.sum().clamp_min(eps)

        p_mask = pred_bin_batch_idxs == b
        if not p_mask.any():
            ce = -th.sum(t_probs * th.log(th.full_like(t_probs, eps)))
            ent = -th.sum(t_probs * t_probs.clamp_min(eps).log())
            raw_loss = th.clamp(ce - ent, min=0.0)

            item_weight = float(item.get("loss_weight", 1.0))
            if item_weight < 0.0:
                item_weight = 0.0
            raw_loss = raw_loss * item_weight

            loss_cap = float(getattr(pl_module.hparams, "oracle_teacher_bin_loss_cap", 8.0))
            if loss_cap > 0:
                raw_loss = th.clamp(raw_loss, max=loss_cap)

            loss_vec[b] = raw_loss
            valid_mask[b] = True
            continue

        p_bins = pred_bin_idxs[p_mask].long()
        p_probs = pred_bin_probs[p_mask]
        p_probs = p_probs / p_probs.sum().clamp_min(eps)

        # R28C: soft/tolerance-aware bin matching.
        # Exact 0.01-bin matching is too harsh for this task:
        # audit shows model/reference agreement is much better under tolerance.
        tol_bins = int(getattr(pl_module.hparams, "oracle_teacher_bin_tolerance_bins", 1))
        tol_bins = max(tol_bins, 0)

        dist = (t_bins.reshape(-1, 1) - p_bins.reshape(1, -1)).abs()

        if tol_bins <= 0:
            weights = dist.eq(0).to(dtype=p_probs.dtype)
        else:
            # Triangular window:
            # dist=0 -> 1.0
            # dist=1 -> 0.5 when tol_bins=1
            # outside window -> 0
            weights = (1.0 - dist.to(dtype=p_probs.dtype) / float(tol_bins + 1)).clamp_min(0.0)

        p_at_t = (weights * p_probs.reshape(1, -1)).sum(dim=1)
        p_at_t = p_at_t.clamp(min=eps, max=1.0)

        ce = -th.sum(t_probs * p_at_t.log())
        ent = -th.sum(t_probs * t_probs.clamp_min(eps).log())

        raw_loss = th.clamp(ce - ent, min=0.0)

        # R35A: confidence-weighted true-anchored teacher.
        # Older R28 used target_value only for filtering.
        # Here we also allow each teacher item to carry a loss_weight,
        # so high-gain / high-CE samples can contribute more while noisy
        # low-gain references stay weak.
        item_weight = float(item.get("loss_weight", 1.0))
        if item_weight < 0.0:
            item_weight = 0.0
        raw_loss = raw_loss * item_weight

        loss_cap = float(getattr(pl_module.hparams, "oracle_teacher_bin_loss_cap", 8.0))
        if loss_cap > 0:
            raw_loss = th.clamp(raw_loss, max=loss_cap)

        loss_vec[b] = raw_loss
        valid_mask[b] = True

    return loss_vec, valid_mask
