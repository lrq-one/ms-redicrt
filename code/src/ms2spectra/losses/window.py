import torch as th


def true_window_distribution_loss(
    true_mzs,
    true_logprobs,
    true_batch_idxs,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
    batch_size,
    match_tol=0.006,
    sigma=0.002,
    max_pred_per_spec=1024,
    max_target_pred_per_spec=512,
    max_true_per_spec=256,
    min_true_prob=0.0,
    min_target_mass=1.0e-12,
    pred_temperature=1.0,
    chunk_size=2048,
):
    """
    R64: true-window distribution loss.

    Purpose:
      Move probability mass from false candidates to predicted candidates
      that fall within a true m/z window.

    For each predicted candidate i:
      target_i = sum_j true_prob_j * Gaussian(pred_mz_i - true_mz_j)
      only if abs(pred_mz_i - true_mz_j) <= match_tol.

    Then:
      loss = CE(target_distribution, softmax(pred_logprobs))
    """
    device = pred_logprobs.device
    dtype = pred_logprobs.dtype

    if hasattr(batch_size, "detach"):
        bs = int(batch_size.detach().cpu().item())
    else:
        bs = int(batch_size)

    base_zero = pred_logprobs.sum() * 0.0

    loss_vec = th.zeros(bs, dtype=dtype, device=device)
    outside_vec = th.zeros(bs, dtype=dtype, device=device)
    valid_vec = th.zeros(bs, dtype=dtype, device=device)
    support_frac_vec = th.zeros(bs, dtype=dtype, device=device)
    pred_used_vec = th.zeros(bs, dtype=dtype, device=device)

    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        return (
            loss_vec + base_zero,
            outside_vec + base_zero,
            valid_vec.mean(),
            support_frac_vec.mean(),
            pred_used_vec.mean(),
        )

    true_mzs = true_mzs.to(device=device, dtype=dtype)
    true_logprobs = true_logprobs.to(device=device, dtype=dtype)
    true_batch_idxs = true_batch_idxs.to(device=device).long()

    pred_mzs = pred_mzs.to(device=device, dtype=dtype)
    pred_logprobs = pred_logprobs.to(device=device, dtype=dtype)
    pred_batch_idxs = pred_batch_idxs.to(device=device).long()

    match_tol = float(match_tol)
    sigma = max(float(sigma), 1.0e-8)
    max_pred_per_spec = int(max_pred_per_spec)
    max_target_pred_per_spec = int(max_target_pred_per_spec)
    max_true_per_spec = int(max_true_per_spec)
    min_true_prob = float(min_true_prob)
    min_target_mass = float(min_target_mass)
    pred_temperature = max(float(pred_temperature), 1.0e-6)
    chunk_size = max(int(chunk_size), 256)

    true_probs_all = true_logprobs.exp()

    for b in range(bs):
        t_mask = true_batch_idxs == b
        p_mask = pred_batch_idxs == b

        if not bool(t_mask.any()) or not bool(p_mask.any()):
            continue

        t_mz = true_mzs[t_mask]
        t_prob = true_probs_all[t_mask]

        if min_true_prob > 0.0:
            keep_t = t_prob >= min_true_prob
            t_mz = t_mz[keep_t]
            t_prob = t_prob[keep_t]

        if t_mz.numel() == 0:
            continue

        if max_true_per_spec > 0 and t_mz.numel() > max_true_per_spec:
            t_prob, t_order = th.topk(t_prob, k=max_true_per_spec, largest=True)
            t_mz = t_mz[t_order]

        p_mz = pred_mzs[p_mask]
        p_score = pred_logprobs[p_mask]

        n_pred = int(p_mz.numel())
        if n_pred == 0:
            continue

        target_raw = th.zeros(n_pred, dtype=dtype, device=device)

        for s in range(0, n_pred, chunk_size):
            e = min(s + chunk_size, n_pred)

            diff = p_mz[s:e].reshape(-1, 1) - t_mz.reshape(1, -1)
            in_win = diff.abs() <= match_tol

            if not bool(in_win.any()):
                continue

            kernel = th.exp(-0.5 * (diff / sigma).pow(2))
            kernel = kernel * in_win.to(dtype=dtype)
            target_raw[s:e] = (kernel * t_prob.reshape(1, -1)).sum(dim=1)

        if float(target_raw.sum().detach().cpu()) <= min_target_mass:
            continue

        keep = th.zeros(n_pred, dtype=th.bool, device=device)

        # Keep high-score predicted candidates as hard negatives.
        if max_pred_per_spec > 0 and n_pred > max_pred_per_spec:
            top_pred = th.topk(p_score, k=max_pred_per_spec, largest=True).indices
            keep[top_pred] = True
        else:
            keep[:] = True

        # Always keep target-positive candidates.
        pos_idx = th.nonzero(target_raw > 0, as_tuple=False).reshape(-1)
        if pos_idx.numel() > 0:
            if max_target_pred_per_spec > 0 and pos_idx.numel() > max_target_pred_per_spec:
                top_pos_rel = th.topk(
                    target_raw[pos_idx],
                    k=max_target_pred_per_spec,
                    largest=True,
                ).indices
                keep[pos_idx[top_pos_rel]] = True
            else:
                keep[pos_idx] = True

        target = target_raw[keep]
        score = p_score[keep]

        target_sum = target.sum()
        if float(target_sum.detach().cpu()) <= min_target_mass:
            continue

        target = target / target_sum.clamp_min(1.0e-12)

        logp = th.log_softmax(score / pred_temperature, dim=0)
        prob = logp.exp()

        dist_loss = -(target.detach() * logp).sum()
        outside_mass = prob[target <= 0].sum()

        loss_vec[b] = dist_loss
        outside_vec[b] = outside_mass
        valid_vec[b] = 1.0
        support_frac_vec[b] = (target > 0).to(dtype=dtype).mean()
        pred_used_vec[b] = float(int(keep.sum().detach().cpu().item()))

    return (
        loss_vec + base_zero,
        outside_vec + base_zero,
        valid_vec.mean(),
        support_frac_vec.mean(),
        pred_used_vec.mean(),
    )
