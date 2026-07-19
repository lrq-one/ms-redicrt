import torch as th
import torch.nn.functional as F


def candidate_presence_rank_loss(
    true_mzs,
    true_logprobs,
    true_batch_idxs,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
    mz_max: float,
    bin_res: float = 0.01,
    topk: int = 100,
    max_pos_per_spec: int = 32,
    max_neg_per_spec: int = 64,
    margin: float = 0.0,
    match_tolerance_abs: float = 0.01,
    min_true_prob: float = 0.001,
    pos_weight_gamma: float = 0.5,
):
    """
    R4c-v2: tolerance-aware, intensity-weighted candidate ranking loss.

    Compared with v1:
    1. positive/negative is decided by abs m/z tolerance, not floor-bin equality.
    2. positives are weighted by matched true intensity.
    3. weak true peaks can be filtered by min_true_prob.
    4. topK/max pairs are smaller to avoid noisy pair explosion.

    Return:
        per-spectrum loss vector, shape [batch_size].
    """
    device = pred_logprobs.device
    dtype = pred_logprobs.dtype

    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        return pred_logprobs.sum().reshape(1) * 0.0

    true_mzs = true_mzs.to(device=device, dtype=dtype)
    pred_mzs = pred_mzs.to(device=device, dtype=dtype)
    true_logprobs = true_logprobs.to(device=device, dtype=dtype)
    pred_logprobs = pred_logprobs.to(device=device, dtype=dtype)

    true_batch_idxs = true_batch_idxs.to(device=device).long()
    pred_batch_idxs = pred_batch_idxs.to(device=device).long()

    batch_size = int(
        max(
            int(true_batch_idxs.max().detach().cpu().item()),
            int(pred_batch_idxs.max().detach().cpu().item()),
        )
    ) + 1

    true_probs = true_logprobs.exp()
    loss_vec = th.zeros(batch_size, dtype=dtype, device=device)

    tol = float(match_tolerance_abs)
    topk = int(topk)
    max_pos_per_spec = int(max_pos_per_spec)
    max_neg_per_spec = int(max_neg_per_spec)
    margin = float(margin)
    min_true_prob = float(min_true_prob)
    pos_weight_gamma = float(pos_weight_gamma)

    for b in range(batch_size):
        t_mask = true_batch_idxs == b
        p_mask = pred_batch_idxs == b

        if not bool(t_mask.any()) or not bool(p_mask.any()):
            continue

        t_mz = true_mzs[t_mask]
        t_prob = true_probs[t_mask]

        # Filter very weak true peaks. They are noisy and should not dominate ranking.
        if min_true_prob > 0.0:
            keep_t = t_prob >= min_true_prob
            t_mz = t_mz[keep_t]
            t_prob = t_prob[keep_t]

        if t_mz.numel() == 0:
            continue

        p_mz = pred_mzs[p_mask]
        p_scores = pred_logprobs[p_mask]

        # Focus on candidates that can affect final spectrum ranking.
        if topk > 0 and p_scores.numel() > topk:
            p_scores, order = th.topk(p_scores, k=topk, largest=True)
            p_mz = p_mz[order]

        # Match by absolute m/z tolerance, not floor-bin equality.
        diff = (p_mz.reshape(-1, 1) - t_mz.reshape(1, -1)).abs()
        match_mat = diff <= tol

        pos_mask = match_mat.any(dim=1)
        neg_mask = ~pos_mask

        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            continue

        pos_scores = p_scores[pos_mask]
        neg_scores = p_scores[neg_mask]

        # Matched true intensity for each positive predicted candidate.
        # If one pred candidate matches multiple true peaks, sum their probabilities.
        pos_match_mat = match_mat[pos_mask].to(dtype=dtype)
        pos_weights = (pos_match_mat * t_prob.reshape(1, -1)).sum(dim=1)

        # Remove pathological zero-weight positives.
        keep_pos = pos_weights > 0
        pos_scores = pos_scores[keep_pos]
        pos_weights = pos_weights[keep_pos]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue

        # Keep strongest true-intensity positives.
        if pos_scores.numel() > max_pos_per_spec:
            _, pos_order = th.topk(pos_weights, k=max_pos_per_spec, largest=True)
            pos_scores = pos_scores[pos_order]
            pos_weights = pos_weights[pos_order]

        # Hard negatives: wrong candidates with highest predicted logprob.
        if neg_scores.numel() > max_neg_per_spec:
            neg_scores, _ = th.topk(neg_scores, k=max_neg_per_spec, largest=True)

        # Weight positives by true intensity, but smooth with gamma.
        pos_weights = pos_weights.clamp_min(1e-12).pow(pos_weight_gamma)
        pos_weights = pos_weights / pos_weights.sum().clamp_min(1e-12)

        pair_loss = F.softplus(
            neg_scores.reshape(1, -1) - pos_scores.reshape(-1, 1) + margin
        )

        per_pos_loss = pair_loss.mean(dim=1)
        loss_vec[b] = (pos_weights * per_pos_loss).sum()

    return loss_vec
