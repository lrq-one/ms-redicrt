from __future__ import annotations
import numpy as np
import torch as th
from torch.nn import functional as F
from ms2spectra.utils.misc_utils import scatter_logsumexp

def get_batch_size(batch):
    bs = batch['batch_size']
    if hasattr(bs, 'detach'):
        return int(bs.detach().cpu().item())
    return int(bs)

def get_spec_ce(batch, bs):
    if 'spec_ce' not in batch:
        return np.zeros(bs, dtype=np.float32)
    ce = batch['spec_ce']
    if isinstance(ce, th.Tensor):
        ce_np = ce.detach().float().reshape(-1).cpu().numpy()
    else:
        ce_np = np.asarray(ce, dtype=np.float32).reshape(-1)
    if len(ce_np) == bs:
        return ce_np.astype(np.float32)
    if 'spec_ce_batch_idxs' in batch:
        idx = batch['spec_ce_batch_idxs'].detach().long().cpu().numpy()
        out = np.zeros(bs, dtype=np.float32)
        cnt = np.zeros(bs, dtype=np.float32)
        for (c, b) in zip(ce_np, idx):
            if 0 <= int(b) < bs:
                out[int(b)] += float(c)
                cnt[int(b)] += 1.0
        return (out / np.maximum(cnt, 1.0)).astype(np.float32)
    return np.resize(ce_np, bs).astype(np.float32)

def ce_weight_tensor(ce_np, device, low_w, mid_w, high_w):
    w = np.ones_like(ce_np, dtype=np.float32)
    w[ce_np <= 20] = float(low_w)
    w[(ce_np > 20) & (ce_np <= 40)] = float(mid_w)
    w[ce_np > 40] = float(high_w)
    return th.tensor(w, dtype=th.float32, device=device)

def scatter_add_1d(x, idx, dim_size):
    out = th.zeros(dim_size, dtype=x.dtype, device=x.device)
    if x.numel() > 0:
        out.index_add_(0, idx.long(), x)
    return out

def formula_losses(raw, target_formula, ce_np, low_w, mid_w, high_w, kl_weight, rank_weight, false_weight, target_topk, neg_topk, margin, neg_target_max):
    pred_logp = raw['pred_formula_logprobs']
    f_bidx = raw['pred_formula_batch_idxs'].long()
    bs = int(f_bidx.max().detach().cpu().item()) + 1
    t = target_formula.to(pred_logp.device)
    t_mass = scatter_add_1d(t, f_bidx, bs)
    valid = t_mass > 1e-08
    t_norm = t / th.clamp(t_mass[f_bidx], min=1e-08)
    ce_w = ce_weight_tensor(ce_np, pred_logp.device, low_w=low_w, mid_w=mid_w, high_w=high_w)
    if ce_w.shape[0] != bs:
        ce_w = th.ones(bs, dtype=th.float32, device=pred_logp.device)
    per_formula_ce = -t_norm * pred_logp
    per_spec_ce = scatter_add_1d(per_formula_ce, f_bidx, bs)
    if valid.any():
        kl_loss = (per_spec_ce[valid] * ce_w[valid]).sum() / th.clamp(ce_w[valid].sum(), min=1e-08)
    else:
        kl_loss = pred_logp.sum() * 0.0
    rank_parts = []
    false_parts = []
    rank_weights = []
    false_weights = []
    n_rank_specs = 0
    n_false_specs = 0
    for b in range(bs):
        idx = th.nonzero(f_bidx == b, as_tuple=False).reshape(-1)
        if idx.numel() < 2 or not bool(valid[b].detach().cpu().item()):
            continue
        ti = t_norm[idx]
        lp = pred_logp[idx]
        pp = lp.exp()
        pos_mask = ti > 0
        pos_rel = th.nonzero(pos_mask, as_tuple=False).reshape(-1)
        if pos_rel.numel() == 0:
            continue
        k_pos = min(int(target_topk), int(pos_rel.numel()))
        pos_score = ti[pos_rel]
        pos_rel = pos_rel[th.topk(pos_score, k=k_pos, largest=True).indices]
        neg_mask = ti <= float(neg_target_max)
        neg_rel = th.nonzero(neg_mask, as_tuple=False).reshape(-1)
        if neg_rel.numel() == 0:
            order_low_t = th.argsort(ti, descending=False)
            pos_set = set([int(x) for x in pos_rel.detach().cpu().tolist()])
            fallback = [int(x) for x in order_low_t.detach().cpu().tolist() if int(x) not in pos_set]
            if len(fallback) > 0:
                neg_rel = th.tensor(fallback, dtype=th.long, device=pred_logp.device)
            else:
                continue
        k_neg = min(int(neg_topk), int(neg_rel.numel()))
        neg_rel = neg_rel[th.topk(lp[neg_rel], k=k_neg, largest=True).indices]
        pos_lp = lp[pos_rel]
        neg_lp = lp[neg_rel]
        rank_mat = F.relu(float(margin) - pos_lp[:, None] + neg_lp[None, :])
        rank_l = rank_mat.mean()
        false_l = pp[neg_rel].sum()
        w = ce_w[b]
        rank_parts.append(rank_l * w)
        false_parts.append(false_l * w)
        rank_weights.append(w)
        false_weights.append(w)
        n_rank_specs += 1
        n_false_specs += 1
    if len(rank_parts):
        rank_loss = th.stack(rank_parts).sum() / th.clamp(th.stack(rank_weights).sum(), min=1e-08)
    else:
        rank_loss = pred_logp.sum() * 0.0
    if len(false_parts):
        false_loss = th.stack(false_parts).sum() / th.clamp(th.stack(false_weights).sum(), min=1e-08)
    else:
        false_loss = pred_logp.sum() * 0.0
    total = float(kl_weight) * kl_loss + float(rank_weight) * rank_loss + float(false_weight) * false_loss
    with th.no_grad():
        stat = {'kl_loss': float(kl_loss.detach().cpu().item()), 'rank_loss': float(rank_loss.detach().cpu().item()), 'false_loss': float(false_loss.detach().cpu().item()), 'valid_frac': float(valid.float().mean().detach().cpu().item()), 'target_mass': float(t_mass[valid].mean().detach().cpu().item()) if valid.any() else 0.0, 'n_rank_specs': float(n_rank_specs), 'n_false_specs': float(n_false_specs)}
    return (total, stat)

def build_official_bins_and_raw_contrib_hard(model, batch, raw, bin_res=0.01):
    """
    Rebuild R98 official bins and keep raw contributor:
      raw_mz
      raw_prob
      raw_formula_id
      raw_to_official
    """
    device = next(model.parameters()).device
    bs = get_batch_size(batch)
    hparams = model.hparams
    pred_mzs = raw['pred_mzs']
    pred_logprobs = raw['pred_logprobs']
    pred_batch_idxs = raw['pred_batch_idxs'].long()
    mz_max = float(getattr(hparams, 'mz_max', 1500.0))
    max_bins = int(getattr(hparams, 'binned_spectrum_renderer_max_bins', 0))
    preserve_mass = bool(getattr(hparams, 'binned_spectrum_renderer_preserve_mass', True))
    valid = th.isfinite(pred_mzs) & th.isfinite(pred_logprobs) & (pred_mzs >= 0.0) & (pred_mzs < mz_max)
    pred_mzs_v = pred_mzs[valid]
    pred_logprobs_v = pred_logprobs[valid]
    pred_batch_v = pred_batch_idxs[valid]

    def opt_tensor(key, default=None):
        x = raw.get(key, None)
        if x is None or not isinstance(x, th.Tensor):
            return default
        if x.shape[0] != valid.shape[0]:
            return default
        return x[valid]
    formula_ids = opt_tensor('pred_spec_formula_global_idxs', None)
    if formula_ids is None:
        formula_ids = th.full_like(pred_batch_v, -1)
    else:
        formula_ids = formula_ids.long()
    if pred_mzs_v.numel() == 0:
        return None
    old_lse = scatter_logsumexp(pred_logprobs_v.detach(), pred_batch_v.detach(), dim_size=bs)
    bin_ids = th.round(pred_mzs_v.detach().float() / float(bin_res)).long()
    pair = th.stack([pred_batch_v.detach().long(), bin_ids], dim=1)
    (uniq_pair, inv) = th.unique(pair, dim=0, return_inverse=True)
    merged_logp = scatter_logsumexp(pred_logprobs_v.detach(), inv, dim_size=uniq_pair.shape[0])
    merged_batch = uniq_pair[:, 0].long()
    merged_mz = uniq_pair[:, 1].to(dtype=pred_mzs_v.dtype) * float(bin_res)
    if max_bins > 0:
        keep_parts = []
        for b in range(bs):
            idx = th.nonzero(merged_batch == b, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > max_bins:
                rel = th.topk(merged_logp[idx], k=max_bins, largest=True).indices
                idx = idx[rel]
            keep_parts.append(idx)
        keep = th.cat(keep_parts, dim=0) if len(keep_parts) else th.zeros(0, dtype=th.long, device=device)
    else:
        keep = th.arange(merged_logp.numel(), device=device)
    old_to_final = th.full((merged_logp.numel(),), -1, dtype=th.long, device=device)
    old_to_final[keep] = th.arange(keep.numel(), device=device)
    raw_to_official = old_to_final[inv]
    raw_keep = raw_to_official >= 0
    official_mz = merged_mz[keep]
    official_logp = merged_logp[keep]
    official_batch = merged_batch[keep]
    if preserve_mass and official_logp.numel() > 0:
        new_lse = scatter_logsumexp(official_logp, official_batch, dim_size=bs)
        official_logp = official_logp - new_lse[official_batch] + old_lse[official_batch]
    return {'bs': bs, 'official_mz': official_mz.detach().cpu().numpy().astype(np.float64), 'official_prob': official_logp.detach().exp().cpu().numpy().astype(np.float64), 'official_batch': official_batch.detach().cpu().numpy().astype(np.int64), 'raw_to_official': raw_to_official[raw_keep].detach().cpu().numpy().astype(np.int64), 'raw_mz': pred_mzs_v[raw_keep].detach().cpu().numpy().astype(np.float64), 'raw_prob': pred_logprobs_v[raw_keep].detach().exp().cpu().numpy().astype(np.float64), 'raw_formula_ids': formula_ids[raw_keep].detach().cpu().numpy().astype(np.int64)}

def match_true_peaks_to_official_events(batch, off, tol=0.01):
    """
    Return matched events:
      spec b
      official bin index
      true_mz
      true_intensity
    """
    bs = off['bs']
    official_mz = off['official_mz']
    official_batch = off['official_batch']
    true_mzs = batch['spec_mzs'].detach().cpu().numpy().astype(np.float64)
    true_ints = batch['spec_ints'].detach().cpu().numpy().astype(np.float64)
    true_batch = batch['spec_batch_idxs'].detach().cpu().numpy().astype(np.int64)
    events = []
    matched_true_count = np.zeros(bs, dtype=np.float64)
    true_peak_count = np.zeros(bs, dtype=np.float64)
    for b in range(bs):
        pi = np.where(official_batch == b)[0]
        ti = np.where(true_batch == b)[0]
        if len(pi) == 0 or len(ti) == 0:
            continue
        pmz = official_mz[pi]
        tmz = true_mzs[ti]
        tint = true_ints[ti]
        ok = np.isfinite(tmz) & np.isfinite(tint) & (tint > 0)
        tmz = tmz[ok]
        tint = tint[ok]
        if len(tmz) == 0:
            continue
        true_peak_count[b] = len(tmz)
        sort_p = np.argsort(pmz)
        pmz_s = pmz[sort_p]
        pi_s = pi[sort_p]
        for (mz, inten) in zip(tmz, tint):
            pos = np.searchsorted(pmz_s, mz)
            best_j = -1
            best_d = 1000000000.0
            for jj in [pos - 1, pos, pos + 1]:
                if 0 <= jj < len(pmz_s):
                    d = abs(float(pmz_s[jj]) - float(mz))
                    if d < best_d:
                        best_d = d
                        best_j = jj
            if best_j >= 0 and best_d <= tol:
                oi = int(pi_s[best_j])
                events.append((b, oi, float(mz), float(inten)))
                matched_true_count[b] += 1.0
    support_recall = matched_true_count / np.maximum(true_peak_count, 1.0)
    return (events, support_recall)

def build_target_formula_hard(model, batch, raw, tol=0.01, bin_res=0.01, mz_sigma=0.003, hard_formula_topk=3, formula_score_mode='max', prob_alpha=0.0):
    """
    Harder formula target.

    Difference from R143:
      R143 distributes target intensity by current raw probability.
      R144 distributes target intensity by m/z proximity to the true peak.

    For each matched true peak:
      official bin -> raw contributors inside that bin
      formula score = max/sum exp(-(raw_mz-true_mz)^2/(2*sigma^2))
      optional tiny tie-breaker: raw_prob^prob_alpha
      keep formula topK
    """
    off = build_official_bins_and_raw_contrib_hard(model, batch, raw, bin_res=bin_res)
    if off is None:
        return None
    (events, support_recall) = match_true_peaks_to_official_events(batch, off, tol=tol)
    n_formula = int(raw['pred_formula_logprobs'].shape[0])
    target_formula = np.zeros(n_formula, dtype=np.float32)
    raw_to_off = off['raw_to_official']
    raw_mz = off['raw_mz']
    raw_prob = off['raw_prob']
    raw_formula = off['raw_formula_ids']
    sigma = max(float(mz_sigma), 1e-06)
    k = max(int(hard_formula_topk), 1)
    for (b, off_idx, true_mz, true_int) in events:
        rows = np.where(raw_to_off == off_idx)[0]
        if len(rows) == 0:
            continue
        fids = raw_formula[rows]
        mzs = raw_mz[rows]
        probs = raw_prob[rows]
        valid = (fids >= 0) & (fids < n_formula) & np.isfinite(mzs)
        if not valid.any():
            continue
        fids = fids[valid]
        mzs = mzs[valid]
        probs = probs[valid]
        raw_scores = np.exp(-0.5 * ((mzs - true_mz) / sigma) ** 2)
        if float(prob_alpha) > 0:
            p = np.maximum(probs, 1e-12)
            raw_scores = raw_scores * p ** float(prob_alpha)
        formula_score = {}
        for (fid, sc) in zip(fids.tolist(), raw_scores.tolist()):
            fid = int(fid)
            sc = float(sc)
            if formula_score_mode == 'sum':
                formula_score[fid] = formula_score.get(fid, 0.0) + sc
            elif formula_score_mode == 'max':
                formula_score[fid] = max(formula_score.get(fid, 0.0), sc)
            else:
                raise ValueError(f'Unknown formula_score_mode={formula_score_mode}')
        if not formula_score:
            continue
        items = sorted(formula_score.items(), key=lambda x: x[1], reverse=True)
        items = items[:min(k, len(items))]
        scores = np.asarray([max(v, 1e-12) for (_, v) in items], dtype=np.float64)
        scores = scores / max(scores.sum(), 1e-12)
        for ((fid, _), w) in zip(items, scores):
            target_formula[int(fid)] += float(true_int) * float(w)
    target_formula_t = th.tensor(target_formula, dtype=raw['pred_formula_logprobs'].dtype, device=raw['pred_formula_logprobs'].device)
    support_recall_t = th.tensor(support_recall, dtype=th.float32, device=raw['pred_formula_logprobs'].device)
    return {'target_formula': target_formula_t, 'support_recall': support_recall_t, 'n_events': len(events), 'n_official_bins': float(len(off['official_mz']))}
