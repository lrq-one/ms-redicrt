import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from ms2spectra.workflow import load_config, init_dataset, init_dataloader
from ms2spectra.training import FragGNNPL


def move_to_device(obj, device):
    if isinstance(obj, th.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    if hasattr(obj, "to") and not isinstance(obj, (str, bytes)):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def load_state_dict_any(path):
    ckpt = th.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def force_r160_arch(cfg):
    cfg.setdefault("frag_params", {})
    cfg["frag_params"]["formula_comp_feats"] = True
    cfg["frag_params"]["formula_comp_feat_size"] = 18

    cfg["use_formula_comp_residual"] = True
    cfg["formula_comp_hidden_size"] = 128
    cfg["formula_comp_dropout"] = 0.05
    cfg["formula_comp_delta_scale"] = 0.05
    cfg["formula_comp_center_per_spectrum"] = True
    cfg["formula_comp_feat_size"] = 18

    cfg["use_ce_response_scorer"] = True
    cfg["ce_response_hidden_size"] = 128
    cfg["ce_response_dropout"] = 0.05
    cfg["ce_response_delta_scale"] = 0.020
    cfg["ce_response_use_formula_comp"] = True
    cfg["ce_response_use_depth"] = True
    cfg["ce_response_use_h"] = True

    cfg["use_ce_flowfrag"] = True
    cfg["ce_flowfrag_lambda_max"] = 0.15
    cfg["ce_flowfrag_hidden_size"] = 128
    cfg["ce_flowfrag_dropout"] = 0.05
    cfg["ce_flowfrag_max_depth"] = 4
    cfg["ce_flowfrag_mixture_hidden_size"] = 128
    cfg["ce_flowfrag_mixture_dropout"] = 0.05
    cfg["ce_flowfrag_mixture_init_bias"] = -3.0
    cfg["ce_flowfrag_delta_clip"] = 3.0
    cfg["ce_flowfrag_use_direct_node"] = True
    cfg["ce_flowfrag_direct_mix"] = 0.35

    cfg["use_binned_spectrum_renderer"] = True
    cfg["binned_spectrum_renderer_apply_train"] = True
    cfg["binned_spectrum_renderer_bin_res"] = 0.01
    cfg["binned_spectrum_renderer_preserve_mass"] = True
    cfg["binned_spectrum_renderer_max_bins"] = 0

    cfg["use_ce_weighted_binned_aux_loss"] = True
    cfg["ce_binned_aux_loss_weight"] = 0.0015
    cfg["ce_binned_aux_mid_threshold"] = 20.0
    cfg["ce_binned_aux_high_threshold"] = 40.0
    cfg["ce_binned_aux_low_weight"] = 0.25
    cfg["ce_binned_aux_mid_weight"] = 1.75
    cfg["ce_binned_aux_high_weight"] = 2.25

    cfg["use_r117_false_mass_aux_loss"] = False
    cfg["r117_weight"] = 0.0
    return cfg


def find_ce(batch):
    for k, v in batch.items():
        if isinstance(v, th.Tensor):
            lk = str(k).lower()
            if ("ce" in lk or "collision" in lk or "energy" in lk or lk == "ace") and v.numel() > 0:
                return v.reshape(-1).float(), k
    raise RuntimeError("cannot find CE tensor in batch")


def ce_bucket_names(ce):
    out = []
    for x in ce.detach().cpu().numpy().reshape(-1):
        if x <= 20:
            out.append("low_<=20")
        elif x <= 40:
            out.append("mid_20_40")
        else:
            out.append("high_>40")
    return out


def ce_weight_values(ce, low_w, mid_w, high_w):
    w = th.ones_like(ce, dtype=th.float)
    w = th.where(ce <= 20, th.full_like(w, float(low_w)), w)
    w = th.where((ce > 20) & (ce <= 40), th.full_like(w, float(mid_w)), w)
    w = th.where(ce > 40, th.full_like(w, float(high_w)), w)
    return w / w.mean().clamp_min(1e-12)


def dense_by_round_bins(mzs, vals, batch_idxs, batch_size, mz_max, bin_res):
    device = mzs.device
    dtype = vals.dtype
    num_bins = int(float(mz_max) / float(bin_res)) + 2
    dense = th.zeros((batch_size, num_bins), device=device, dtype=dtype)

    if mzs.numel() == 0:
        return dense

    mzs = th.nan_to_num(mzs, nan=0.0, posinf=float(mz_max), neginf=0.0)
    vals = th.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    bins = th.round(mzs / float(bin_res)).long().clamp(0, num_bins - 1)
    bidx = batch_idxs.long().clamp(0, batch_size - 1)
    flat = bidx * num_bins + bins

    dense_flat = dense.reshape(-1)
    dense_flat.index_add_(0, flat, vals)
    return dense_flat.reshape(batch_size, num_bins)


def cosine_dense(a, b, eps=1e-12):
    return (a * b).sum(1) / (
        a.pow(2).sum(1).sqrt() * b.pow(2).sum(1).sqrt()
    ).clamp_min(eps)


def jss_dense(a, b, eps=1e-12):
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a = a / a.sum(1, keepdim=True).clamp_min(eps)
    b = b / b.sum(1, keepdim=True).clamp_min(eps)
    m = 0.5 * (a + b)
    jsd = 0.5 * (
        (a * ((a + eps).log() - (m + eps).log())).sum(1)
        + (b * ((b + eps).log() - (m + eps).log())).sum(1)
    )
    return 1.0 - jsd / math.log(2.0)


def rank_feature(prob, bidx, batch_size):
    out = th.zeros_like(prob)
    for i in range(batch_size):
        loc = th.nonzero(bidx == i, as_tuple=False).reshape(-1)
        if loc.numel() == 0:
            continue
        order = th.argsort(prob[loc], descending=True)
        n = loc.numel()
        if n <= 1:
            out[loc] = 0.0
        else:
            out[loc[order]] = th.arange(n, device=prob.device, dtype=prob.dtype) / float(n - 1)
    return out


def local_density_features(pred_mz, prob, bidx, batch_size, mz_max, bin_res):
    device = pred_mz.device
    dtype = pred_mz.dtype
    num_bins = int(float(mz_max) / float(bin_res)) + 2

    bins = th.round(pred_mz / float(bin_res)).long().clamp(0, num_bins - 1)
    flat = bidx.long().clamp(0, batch_size - 1) * num_bins + bins

    count_dense = th.zeros(batch_size * num_bins, device=device, dtype=dtype)
    mass_dense = th.zeros(batch_size * num_bins, device=device, dtype=dtype)

    count_dense.index_add_(0, flat, th.ones_like(prob))
    mass_dense.index_add_(0, flat, prob)

    count_dense = count_dense.reshape(batch_size, num_bins)
    mass_dense = mass_dense.reshape(batch_size, num_bins)

    def gather_window(radius):
        c = th.zeros_like(prob)
        m = th.zeros_like(prob)
        for off in range(-radius, radius + 1):
            idx = (bins + off).clamp(0, num_bins - 1)
            c = c + count_dense[bidx, idx]
            m = m + mass_dense[bidx, idx]
        return c, m

    c1, m1 = gather_window(1)
    c3, m3 = gather_window(3)
    c5, m5 = gather_window(5)

    feats = th.stack([
        th.log1p(c1) / 4.0,
        th.log1p(c3) / 4.0,
        th.log1p(c5) / 4.0,
        th.log1p(m1 * 1000.0) / 8.0,
        th.log1p(m3 * 1000.0) / 8.0,
        th.log1p(m5 * 1000.0) / 8.0,
        (prob / m1.clamp_min(1e-12)).clamp(0, 10) / 10.0,
        (prob / m5.clamp_min(1e-12)).clamp(0, 10) / 10.0,
    ], dim=1)

    return feats






def attach_raw_rich_features(base, batch, results, max_extra_dims=96, bin_res=0.01):
    """
    R173A: richer internal candidate features.

    Raw forward output is R54-expanded. _common_step results are final binned peaks.
    We build raw-entry features first, then aggregate raw entries to final result entries
    by (batch_idx, mz_bin).

    Feature sources:
      A) old/raw peak-entry features from R172D
      B) formula-level logprob gathered by pred_spec_formula_global_idxs
      C) joint-level features aggregated to formula, then gathered to peak
      D) node-level features aggregated through joint->formula, then gathered to peak
    """
    try:
        if not isinstance(results, dict):
            return results
        if "pred_mzs" not in results or "pred_logprobs" not in results or "pred_batch_idxs" not in results:
            return results

        res_mz = results["pred_mzs"]
        res_logp = results["pred_logprobs"]
        res_bidx = results["pred_batch_idxs"].long()

        n_res = int(res_mz.numel())
        if n_res <= 0:
            return results

        raw = base.forward(**batch)
        if not isinstance(raw, dict):
            return results
        if "pred_mzs" not in raw or "pred_logprobs" not in raw or "pred_batch_idxs" not in raw:
            return results

        device = res_mz.device
        dtype = res_logp.dtype

        raw_mz = raw["pred_mzs"].to(device)
        raw_logp = raw["pred_logprobs"].to(device)
        raw_bidx = raw["pred_batch_idxs"].to(device).long()

        n_raw = int(raw_mz.numel())
        if n_raw <= 0:
            return results

        max_total = int(max_extra_dims)
        raw_cols = []
        names = []
        total = 0

        # raw R54 entry -> old peak-entry index
        offset_idx = None
        oi = raw.get("pred_spec_offset_group_idxs", None)
        if isinstance(oi, th.Tensor) and int(oi.numel()) == n_raw:
            offset_idx = oi.to(device).long()

        def _scale_feature(key, x):
            lk = str(key).lower()
            x = th.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)

            if "logprob" in lk or "logit" in lk or "delta" in lk or "logsum" in lk:
                return x.clamp(-30.0, 10.0) / 10.0
            if "comp_feats" in lk:
                return x.clamp(-10.0, 10.0)
            if "r12" in lk:
                return x.clamp(-10.0, 10.0)
            if "channel" in lk or "h_count" in lk or "depth" in lk:
                return x.clamp(-20.0, 20.0) / 20.0
            if "mask" in lk:
                return x.clamp(0.0, 1.0)

            return x.clamp(-20.0, 20.0) / 20.0

        def _append_raw_entry_feature(key, v, use_offset=True, max_take=16):
            nonlocal total, raw_cols, names

            if total >= max_total:
                return
            if not isinstance(v, th.Tensor):
                return
            if v.numel() == 0 or v.dim() == 0:
                return

            x = v.to(device).float()

            # Direct raw R54-expanded entry feature.
            if int(x.shape[0]) == n_raw:
                pass
            # Old peak-entry feature. Expand old -> raw by R54 offset index.
            elif use_offset and offset_idx is not None and int(x.shape[0]) > int(offset_idx.max().item()):
                x = x[offset_idx]
            else:
                return

            if x.dim() == 1:
                x = x.reshape(n_raw, 1)
            else:
                x = x.reshape(n_raw, -1)

            take = min(int(x.shape[1]), int(max_take), max_total - total)
            if take <= 0:
                return

            x = _scale_feature(key, x[:, :take])
            raw_cols.append(x.to(dtype=dtype))
            names.append(f"{key}:{take}")
            total += take

        # -------- A) R172D stable peak-entry features --------
        stable_peak_keys = [
            "r172_peak_rich_feats",
            "pred_spec_formula_logprobs",
            "pred_spec_formula_comp_feats",
            "pred_spec_base_peak_logprobs",
            "pred_spec_peak_logprobs",
            "pred_spec_peak_channels",
            "pred_rendered_peak_gate_logits",
            "pred_rendered_peak_gate_delta",
            "pred_refiner_delta",
            "pred_refiner_delta_valid_mask",
        ]

        for key in stable_peak_keys:
            if total >= max_total:
                break
            if key in raw:
                max_take = 18 if key == "pred_spec_formula_comp_feats" else 8
                _append_raw_entry_feature(key, raw[key], use_offset=True, max_take=max_take)

        # Build raw-entry formula index via spec_formula_global_idxs.
        raw_formula_idx = None
        sf = raw.get("pred_spec_formula_global_idxs", None)
        if isinstance(sf, th.Tensor) and sf.numel() > 0 and sf.dim() > 0:
            sf = sf.to(device).long()
            if int(sf.shape[0]) == n_raw:
                raw_formula_idx = sf
            elif offset_idx is not None and int(sf.shape[0]) > int(offset_idx.max().item()):
                raw_formula_idx = sf[offset_idx]

        n_formula = 0
        if raw_formula_idx is not None and raw_formula_idx.numel() > 0:
            n_formula = int(raw_formula_idx.max().item()) + 1
        flp = raw.get("pred_formula_logprobs", None)
        if isinstance(flp, th.Tensor) and flp.dim() > 0:
            n_formula = max(n_formula, int(flp.shape[0]))

        def _append_formula_feature(key, formula_feat, max_take=8):
            nonlocal total, raw_cols, names

            if total >= max_total:
                return
            if raw_formula_idx is None:
                return
            if not isinstance(formula_feat, th.Tensor):
                return
            if formula_feat.numel() == 0 or formula_feat.dim() == 0:
                return

            x = formula_feat.to(device).float()
            if x.dim() == 1:
                x = x.reshape(-1, 1)
            else:
                x = x.reshape(int(x.shape[0]), -1)

            if int(x.shape[0]) <= int(raw_formula_idx.max().item()):
                return

            x = x[raw_formula_idx]
            if int(x.shape[0]) != n_raw:
                return

            take = min(int(x.shape[1]), int(max_take), max_total - total)
            if take <= 0:
                return

            x = _scale_feature(key, x[:, :take])
            raw_cols.append(x.to(dtype=dtype))
            names.append(f"{key}:{take}")
            total += take

        # -------- B) formula-level feature --------
        if raw_formula_idx is not None and "pred_formula_logprobs" in raw:
            _append_formula_feature("formula_pred_formula_logprobs", raw["pred_formula_logprobs"], max_take=1)

        # -------- C/D) joint/node aggregated-to-formula features --------
        jf = raw.get("pred_joint_formula_idxs", None)
        jl = raw.get("pred_joint_logprobs", None)

        if raw_formula_idx is not None and n_formula > 0 and isinstance(jf, th.Tensor) and isinstance(jl, th.Tensor):
            if jf.numel() > 0 and jl.numel() > 0 and jf.dim() > 0 and jl.dim() > 0:
                jf = jf.to(device).long().reshape(-1)
                jl = jl.to(device).float().reshape(-1)
                n_joint = min(int(jf.numel()), int(jl.numel()))
                jf = jf[:n_joint]
                jl = jl[:n_joint]

                joint_valid = (jf >= 0) & (jf < int(n_formula))
                if joint_valid.any():
                    jf_v = jf[joint_valid]
                    jl_v = jl[joint_valid]
                    jw_v = jl_v.clamp(-30.0, 5.0).exp().clamp_min(1e-12)

                    denom = th.zeros(int(n_formula), device=device, dtype=jl_v.dtype)
                    denom.scatter_add_(0, jf_v, jw_v)
                    denom_safe = denom.clamp_min(1e-12)

                    # joint logsum per formula
                    joint_sumprob = denom.clamp_min(1e-12).log()
                    _append_formula_feature("agg_joint_logsum_logprob", joint_sumprob, max_take=1)

                    # joint weighted mean logprob per formula
                    mean_jlp = th.zeros(int(n_formula), device=device, dtype=jl_v.dtype)
                    mean_jlp.scatter_add_(0, jf_v, jw_v * jl_v)
                    mean_jlp = mean_jlp / denom_safe
                    _append_formula_feature("agg_joint_mean_logprob", mean_jlp, max_take=1)

                    def _joint_weighted_formula_avg(name, joint_values, max_take=8, valid_override=None):
                        if not isinstance(joint_values, th.Tensor):
                            return
                        if joint_values.numel() == 0 or joint_values.dim() == 0:
                            return

                        v = joint_values.to(device).float()
                        if v.dim() == 1:
                            v = v.reshape(-1, 1)
                        else:
                            v = v.reshape(int(v.shape[0]), -1)

                        n_use = min(int(v.shape[0]), n_joint)
                        v = v[:n_use]
                        jf_local = jf[:n_use]
                        jl_local = jl[:n_use]
                        valid = (jf_local >= 0) & (jf_local < int(n_formula))
                        if valid_override is not None:
                            valid = valid & valid_override[:n_use].to(device).bool()

                        if not valid.any():
                            return

                        f = jf_local[valid]
                        lp = jl_local[valid]
                        w = lp.clamp(-30.0, 5.0).exp().clamp_min(1e-12)
                        vv = v[valid]

                        d = int(vv.shape[1])
                        out = th.zeros((int(n_formula), d), device=device, dtype=vv.dtype)
                        den = th.zeros(int(n_formula), device=device, dtype=vv.dtype)

                        out.scatter_add_(0, f[:, None].expand(-1, d), vv * w[:, None])
                        den.scatter_add_(0, f, w)

                        out = out / den.clamp_min(1e-12).unsqueeze(1)
                        _append_formula_feature(name, out, max_take=max_take)

                    # joint h-count expectation
                    if "pred_joint_h_counts" in raw:
                        _joint_weighted_formula_avg("agg_joint_h_counts", raw["pred_joint_h_counts"], max_take=1)

                        h = raw["pred_joint_h_counts"]
                        if isinstance(h, th.Tensor):
                            _joint_weighted_formula_avg("agg_joint_abs_h_counts", h.float().abs(), max_take=1)

                    # R12 refiner internal features at joint level
                    if "pred_joint_r12_feats" in raw:
                        _joint_weighted_formula_avg("agg_joint_r12_feats", raw["pred_joint_r12_feats"], max_take=8)

                    # node features through joint_node_idxs
                    jn = raw.get("pred_joint_node_idxs", None)
                    if isinstance(jn, th.Tensor) and jn.numel() > 0:
                        jn = jn.to(device).long().reshape(-1)
                        n_use = min(int(jn.numel()), n_joint)
                        jn_use = jn[:n_use]

                        node_lp = raw.get("pred_node_logprobs", None)
                        if isinstance(node_lp, th.Tensor) and node_lp.dim() > 0 and node_lp.numel() > 0:
                            node_lp = node_lp.to(device).float().reshape(-1)
                            valid_node = (jn_use >= 0) & (jn_use < int(node_lp.numel()))
                            vals = th.zeros(n_use, device=device, dtype=node_lp.dtype)
                            vals[valid_node] = node_lp[jn_use[valid_node]]
                            _joint_weighted_formula_avg("agg_node_logprobs_by_joint", vals, max_take=1, valid_override=valid_node)

                        node_depth = raw.get("pred_node_depths", None)
                        if isinstance(node_depth, th.Tensor) and node_depth.dim() > 0 and node_depth.numel() > 0:
                            node_depth = node_depth.to(device).float().reshape(-1)
                            valid_node = (jn_use >= 0) & (jn_use < int(node_depth.numel()))
                            vals = th.zeros(n_use, device=device, dtype=node_depth.dtype)
                            vals[valid_node] = node_depth[jn_use[valid_node]]
                            _joint_weighted_formula_avg("agg_node_depths_by_joint", vals, max_take=1, valid_override=valid_node)

                        node_formula_lp = raw.get("pred_node_formula_logprobs", None)
                        if isinstance(node_formula_lp, th.Tensor) and node_formula_lp.dim() > 0 and node_formula_lp.numel() > 0:
                            node_formula_lp = node_formula_lp.to(device).float().reshape(-1)
                            valid_node = (jn_use >= 0) & (jn_use < int(node_formula_lp.numel()))
                            vals = th.zeros(n_use, device=device, dtype=node_formula_lp.dtype)
                            vals[valid_node] = node_formula_lp[jn_use[valid_node]]
                            _joint_weighted_formula_avg("agg_node_formula_logprobs_by_joint", vals, max_take=1, valid_override=valid_node)

        if len(raw_cols) == 0:
            if not hasattr(base, "_r173a_empty_printed"):
                print(
                    "[R173A] no raw-entry rich features built;",
                    "n_res=", n_res,
                    "n_raw=", n_raw,
                    "has_offset=", offset_idx is not None,
                    "has_raw_formula_idx=", raw_formula_idx is not None,
                    "raw_keys_head=", list(raw.keys())[:80],
                )
                base._r173a_empty_printed = True
            return results

        raw_feat = th.cat(raw_cols, dim=1)
        raw_feat = th.nan_to_num(raw_feat, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        d = int(raw_feat.shape[1])

        # Aggregate raw R54 entries to final _common_step entries by (batch_idx, mz_bin).
        br = float(bin_res)
        if br <= 0:
            br = 0.01

        raw_bin = th.round(raw_mz.float() / br).long()
        res_bin = th.round(res_mz.float() / br).long()

        max_bin = int(th.maximum(raw_bin.max(), res_bin.max()).item()) + 1
        raw_key = raw_bidx.long() * max_bin + raw_bin
        res_key = res_bidx.to(device).long() * max_bin + res_bin

        sorted_res_key, order = th.sort(res_key)
        pos = th.searchsorted(sorted_res_key, raw_key)

        ok = (pos >= 0) & (pos < n_res)
        pos_safe = pos.clamp_min(0).clamp_max(n_res - 1)
        matched_sorted_key = sorted_res_key[pos_safe]
        ok = ok & (matched_sorted_key == raw_key)

        if not ok.any():
            if not hasattr(base, "_r173a_nomatch_printed"):
                print(
                    "[R173A] no raw->result bin matches;",
                    "n_res=", n_res,
                    "n_raw=", n_raw,
                    "bin_res=", br,
                    "raw_bin_range=", (int(raw_bin.min().item()), int(raw_bin.max().item())),
                    "res_bin_range=", (int(res_bin.min().item()), int(res_bin.max().item())),
                )
                base._r173a_nomatch_printed = True
            return results

        raw_to_res = order[pos_safe[ok]]
        raw_w = raw_logp.float().exp().clamp_min(1e-12)[ok]

        denom = th.zeros(n_res, device=device, dtype=raw_feat.dtype)
        denom.scatter_add_(0, raw_to_res, raw_w)

        agg = th.zeros((n_res, d), device=device, dtype=raw_feat.dtype)
        agg.scatter_add_(0, raw_to_res[:, None].expand(-1, d), raw_feat[ok] * raw_w[:, None])
        agg = agg / denom.clamp_min(1e-12).unsqueeze(1)
        agg = th.nan_to_num(agg, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        results["r173_frag_rich_feats"] = agg.to(dtype=dtype)

        if not hasattr(base, "_r173a_printed"):
            matched_frac = float(ok.float().mean().item())
            covered_frac = float((denom > 0).float().mean().item())
            print(
                "[R173A] attached r173_frag_rich_feats",
                tuple(agg.shape),
                "from",
                names[:120],
                "n_res=", n_res,
                "n_raw=", n_raw,
                "matched_raw_frac=", matched_frac,
                "covered_res_frac=", covered_frac,
            )
            base._r173a_printed = True

    except Exception as e:
        if not hasattr(base, "_r173a_failed_printed"):
            print("[R173A] failed:", repr(e))
            base._r173a_failed_printed = True

    return results

def infer_extra_schema(results, max_extra_dims=32):
    n = int(results["pred_mzs"].numel())
    schema = []
    used = 0

    skip_exact = {
        "pred_mzs", "pred_logprobs", "pred_batch_idxs",
        "true_mzs", "true_logprobs", "true_batch_idxs",
        "unique_id", "mean_loss", "loss",
    }

    good_words = [
        "joint", "frag", "formula", "depth", "ce_flow", "response",
        "residual", "score", "logit", "delta", "h"
    ]

    for k, v in results.items():
        if k in skip_exact:
            continue
        if not isinstance(v, th.Tensor):
            continue
        if v.numel() == 0:
            continue
        # skip scalar tensors such as mean_loss / loss
        if v.dim() == 0:
            continue
        if v.shape[0] != n:
            continue
        lk = str(k).lower()
        if not any(w in lk for w in good_words):
            continue
        if "true" in lk or "batch" in lk or "mzs" in lk or "logprob" in lk:
            continue

        if v.dim() == 1:
            dim = 1
        elif v.dim() == 2:
            dim = min(int(v.shape[1]), max_extra_dims - used)
        else:
            continue

        if dim <= 0:
            break

        schema.append((k, dim))
        used += dim
        if used >= max_extra_dims:
            break

    return schema


def extra_features_from_schema(results, schema, n, device):
    cols = []
    for k, dim in schema:
        v = results.get(k, None)
        if (not isinstance(v, th.Tensor)) or v.numel() == 0 or v.dim() == 0 or v.shape[0] != n:
            cols.append(th.zeros((n, dim), device=device))
            continue

        x = v.float()
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        elif x.dim() == 2:
            x = x[:, :dim]
        else:
            x = th.zeros((n, dim), device=device)

        x = th.nan_to_num(x, nan=0.0, posinf=20.0, neginf=-20.0)
        x = x.clamp(-20, 20) / 10.0

        if x.shape[1] < dim:
            pad = th.zeros((n, dim - x.shape[1]), device=device, dtype=x.dtype)
            x = th.cat([x, pad], dim=1)

        cols.append(x)

    if not cols:
        return None
    return th.cat(cols, dim=1)


def candidate_features(results, batch, mz_max, local_bin_res, extra_schema):
    pred_mz = results["pred_mzs"].float()
    logp = results["pred_logprobs"].float()
    prob = logp.exp().clamp_min(0.0)
    bidx = results["pred_batch_idxs"].long()
    n = int(pred_mz.numel())
    batch_size = int(results["unique_id"].numel())
    device = pred_mz.device

    ce, _ = find_ce(batch)
    ce = ce.to(device).reshape(-1).float()
    ce_entry = ce[bidx]

    # ===== R171 mass-aware candidate features =====
    # Add precursor-relative and neutral-loss features.
    # These are critical for fragment plausibility and are not available to the old 23D basic reranker.
    prec = batch.get("spec_prec_mz", None)
    if prec is None:
        prec = th.full_like(ce, float(mz_max))
    else:
        prec = prec.to(device).reshape(-1).float()

    prec_entry = prec[bidx].clamp_min(1e-6)
    loss_mz = prec_entry - pred_mz
    pos_loss_mz = loss_mz.clamp_min(0.0)

    pred_over_prec = (pred_mz / prec_entry).clamp(0.0, 3.0)
    loss_over_prec = (loss_mz / prec_entry).clamp(-2.0, 3.0)

    mz_defect = pred_mz - th.floor(pred_mz)
    loss_defect = pos_loss_mz - th.floor(pos_loss_mz)
    # ===== end R171 mass-aware candidate features =====

    sum_prob = th.zeros(batch_size, device=device, dtype=prob.dtype)
    sum_prob.index_add_(0, bidx, prob)
    rel_prob = prob / sum_prob[bidx].clamp_min(1e-12)

    max_prob = th.zeros(batch_size, device=device, dtype=prob.dtype)
    for i in range(batch_size):
        loc = th.nonzero(bidx == i, as_tuple=False).reshape(-1)
        if loc.numel() > 0:
            max_prob[i] = prob[loc].max()
    peak_ratio = prob / max_prob[bidx].clamp_min(1e-12)

    rfeat = rank_feature(prob, bidx, batch_size)

    base = th.stack([
        pred_mz / float(mz_max),
        th.log1p(pred_mz) / 8.0,
        th.sqrt(pred_mz.clamp_min(0.0)) / math.sqrt(float(mz_max)),
        logp.clamp(-30, 5) / 10.0,
        prob.clamp(0, 1),
        th.log1p(prob * 1e6) / 14.0,
        rel_prob.clamp(0, 1),
        peak_ratio.clamp(0, 1),
        rfeat.clamp(0, 1),
        th.log1p(rfeat * 1000.0) / 7.0,
        ce_entry / 50.0,
        (ce_entry <= 20).float(),
        ((ce_entry > 20) & (ce_entry <= 40)).float(),
        (ce_entry > 40).float(),
        (ce_entry / 50.0) * (pred_mz / float(mz_max)),

        # R171 mass-aware features.
        prec_entry / float(mz_max),
        pred_over_prec,
        loss_over_prec,
        pos_loss_mz / float(mz_max),
        th.log1p(pos_loss_mz) / 8.0,
        mz_defect.clamp(0.0, 1.0),
        loss_defect.clamp(0.0, 1.0),
        (loss_mz >= 0.0).float(),
        (ce_entry / 50.0) * pred_over_prec,
        (ce_entry / 50.0) * loss_over_prec,
        ((pred_mz > 50.0) & (pred_mz < prec_entry)).float(),
    ], dim=1)

    local = local_density_features(
        pred_mz=pred_mz,
        prob=prob,
        bidx=bidx,
        batch_size=batch_size,
        mz_max=mz_max,
        bin_res=local_bin_res,
    )

    extra = extra_features_from_schema(results, extra_schema, n=n, device=device)

    if extra is None:
        feats = th.cat([base, local], dim=1)
    else:
        feats = th.cat([base, local, extra], dim=1)

    feats = th.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    feats = feats.clamp(-10, 10)
    return feats


def build_targets(results, args):
    pred_mz = results["pred_mzs"].float()
    old_logp = results["pred_logprobs"].float()
    pred_prob = old_logp.exp().clamp_min(0.0)
    pred_bidx = results["pred_batch_idxs"].long()

    true_mz = results["true_mzs"].float()
    true_prob = results["true_logprobs"].exp().float().clamp_min(0.0)
    true_bidx = results["true_batch_idxs"].long()

    batch_size = int(results["unique_id"].numel())
    device = pred_mz.device
    mz_max = float(args.mz_max)
    bin_res = float(args.target_bin_res)
    mz_tol = float(args.mz_tol)
    mz_sigma = float(args.mz_sigma)

    true_dense = dense_by_round_bins(
        true_mz, true_prob, true_bidx,
        batch_size=batch_size,
        mz_max=mz_max,
        bin_res=bin_res,
    )

    num_bins = true_dense.shape[1]
    pred_bins0 = th.round(pred_mz / bin_res).long().clamp(0, num_bins - 1)
    k = max(0, int(math.ceil(mz_tol / bin_res)))

    best_score = th.zeros_like(pred_prob)
    best_mass = th.zeros_like(pred_prob)
    best_weight = th.zeros_like(pred_prob)
    best_bin = pred_bins0.clone()

    for off in range(-k, k + 1):
        idx = (pred_bins0 + off).clamp(0, num_bins - 1)
        center = idx.to(pred_mz.dtype) * bin_res
        dist = (pred_mz - center).abs()
        within = dist <= mz_tol
        mass = true_dense[pred_bidx, idx]
        gauss = th.exp(-0.5 * (dist / max(mz_sigma, 1e-12)).pow(2))
        score = mass * gauss * within.float()

        improve = score > best_score
        best_score = th.where(improve, score, best_score)
        best_mass = th.where(improve, mass, best_mass)
        best_weight = th.where(improve, gauss, best_weight)
        best_bin = th.where(improve, idx, best_bin)

    pos = best_score > 0

    target_mass = th.zeros_like(pred_prob)
    if pos.any():
        flat_group = pred_bidx[pos] * num_bins + best_bin[pos]
        denom = th.zeros(batch_size * num_bins, device=device, dtype=pred_prob.dtype)
        denom.index_add_(0, flat_group, best_weight[pos].clamp_min(1e-12))
        target_pos = best_mass[pos] * best_weight[pos] / denom[flat_group].clamp_min(1e-12)
        target_mass[pos] = target_pos

    residual = th.full_like(pred_prob, -float(args.neg_residual))
    residual[pos] = th.log(target_mass[pos].clamp_min(1e-10)) - old_logp[pos]
    residual = residual.clamp(-float(args.residual_clip), float(args.residual_clip))

    return residual, pos.float(), target_mass, pred_prob


def sample_candidate_rows(feats, residual, pos, target_mass, pred_prob, batch, results, args):
    device = feats.device
    n = feats.shape[0]
    bidx = results["pred_batch_idxs"].long()

    ce, _ = find_ce(batch)
    ce = ce.to(device).reshape(-1).float()
    ce_entry = ce[bidx]
    ce_w = ce_weight_values(ce_entry, args.low_w, args.mid_w, args.high_w)

    pos_mask = pos > 0.5
    neg_mask = ~pos_mask

    pos_idx = th.nonzero(pos_mask, as_tuple=False).reshape(-1)

    neg_idx_all = th.nonzero(neg_mask, as_tuple=False).reshape(-1)

    chosen = []
    if pos_idx.numel() > 0:
        chosen.append(pos_idx)

    if neg_idx_all.numel() > 0:
        neg_score = pred_prob[neg_idx_all]
        k_top = min(int(args.neg_topk_per_batch), int(neg_idx_all.numel()))
        if k_top > 0:
            top_local = th.topk(neg_score, k=k_top, largest=True).indices
            chosen.append(neg_idx_all[top_local])

        k_rand = min(int(args.neg_rand_per_batch), int(neg_idx_all.numel()))
        if k_rand > 0:
            perm = th.randperm(neg_idx_all.numel(), device=device)[:k_rand]
            chosen.append(neg_idx_all[perm])

    if not chosen:
        return None

    idx = th.unique(th.cat(chosen, dim=0))

    y = residual[idx]

    w = ce_w[idx] * (
        pos[idx] * float(args.pos_weight)
        + (1.0 - pos[idx]) * (float(args.neg_weight) + float(args.neg_prob_weight) * pred_prob[idx].clamp(0, 1))
    )

    # Intensity-heavy positives matter more.
    w = w * (1.0 + pos[idx] * float(args.pos_intensity_weight) * target_mass[idx].clamp(0, 1))

    return (
        feats[idx].detach().cpu().numpy().astype(np.float32),
        y.detach().cpu().numpy().astype(np.float32),
        w.detach().cpu().numpy().astype(np.float32),
        pos[idx].detach().cpu().numpy().astype(np.float32),
    )


def train_regressor(X, y, w, args):
    backend = args.backend.lower()

    if backend in ["auto", "lightgbm", "lgbm"]:
        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=int(args.n_estimators),
                learning_rate=float(args.gbdt_lr),
                num_leaves=int(args.num_leaves),
                max_depth=int(args.max_depth),
                min_child_samples=int(args.min_child_samples),
                subsample=float(args.subsample),
                colsample_bytree=float(args.colsample_bytree),
                reg_alpha=float(args.reg_alpha),
                reg_lambda=float(args.reg_lambda),
                random_state=int(args.seed),
                n_jobs=int(args.num_workers),
                verbose=-1,
            )
            model.fit(X, y, sample_weight=w)
            print("[R170] trained backend: lightgbm")
            return model, "lightgbm"
        except Exception as e:
            if backend in ["lightgbm", "lgbm"]:
                raise
            print("[R170] lightgbm unavailable, fallback to sklearn HistGradientBoosting:", repr(e))

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=int(args.hgb_iter),
        learning_rate=float(args.hgb_lr),
        max_leaf_nodes=int(args.hgb_leaf_nodes),
        l2_regularization=float(args.hgb_l2),
        random_state=int(args.seed),
    )
    model.fit(X, y, sample_weight=w)
    print("[R170] trained backend: sklearn_hist_gradient_boosting")
    return model, "sklearn_hgb"


def renormalize_logp(old_logp, bidx, score, alpha, batch_size):
    logits = old_logp + float(alpha) * score
    new_logp = th.empty_like(old_logp)

    for i in range(batch_size):
        loc = th.nonzero(bidx == i, as_tuple=False).reshape(-1)
        if loc.numel() == 0:
            continue
        new_logp[loc] = th.log_softmax(logits[loc], dim=0)

    return new_logp


@th.no_grad()
def eval_split(base, regressor, extra_schema, dl, device, args, split, alpha):
    base.eval()
    rows = []

    for batch in tqdm(dl, desc=f"eval {split} alpha={alpha}"):
        batch = move_to_device(batch, device)
        res = base._common_step(batch, split=split, log=False)
        res = attach_raw_rich_features(base, batch, res, max_extra_dims=int(args.max_extra_dims))

        feats = candidate_features(
            res, batch,
            mz_max=float(base.hparams.mz_max),
            local_bin_res=float(args.local_bin_res),
            extra_schema=extra_schema,
        )

        X = feats.detach().cpu().numpy().astype(np.float32)
        score_np = regressor.predict(X).astype(np.float32)
        score_np = np.clip(score_np, -float(args.score_clip), float(args.score_clip))
        score = th.from_numpy(score_np).to(device=device, dtype=res["pred_logprobs"].dtype)

        bidx = res["pred_batch_idxs"].long()
        batch_size = int(res["unique_id"].numel())

        new_logp = renormalize_logp(
            res["pred_logprobs"].float(),
            bidx,
            score,
            alpha=float(alpha),
            batch_size=batch_size,
        )

        true_dense = dense_by_round_bins(
            res["true_mzs"],
            res["true_logprobs"].exp(),
            res["true_batch_idxs"],
            batch_size=batch_size,
            mz_max=float(base.hparams.mz_max),
            bin_res=float(args.eval_bin_res),
        )

        pred_dense = dense_by_round_bins(
            res["pred_mzs"],
            new_logp.exp(),
            res["pred_batch_idxs"],
            batch_size=batch_size,
            mz_max=float(base.hparams.mz_max),
            bin_res=float(args.eval_bin_res),
        )

        cos = cosine_dense(true_dense, pred_dense)
        jss = jss_dense(true_dense, pred_dense)

        ce, _ = find_ce(batch)
        ce_cpu = ce.detach().cpu().reshape(-1)
        buckets = ce_bucket_names(ce_cpu)

        for i, sid in enumerate(res["unique_id"].detach().cpu().reshape(-1).numpy().astype(int)):
            rows.append({
                "spec_id": int(sid),
                "ce": float(ce_cpu[i]),
                "ce_bucket": buckets[i],
                "cos": float(cos[i].detach().cpu()),
                "jss": float(jss[i].detach().cpu()),
            })

    df = pd.DataFrame(rows)

    out = [{
        "ce_bucket": "global",
        "spec_count": int(len(df)),
        "mean_ce": float(df["ce"].mean()),
        "cos": float(df["cos"].mean()),
        "jss": float(df["jss"].mean()),
    }]

    for b in ["low_<=20", "mid_20_40", "high_>40"]:
        sub = df[df["ce_bucket"] == b]
        out.append({
            "ce_bucket": b,
            "spec_count": int(len(sub)),
            "mean_ce": float(sub["ce"].mean()) if len(sub) else 0.0,
            "cos": float(sub["cos"].mean()) if len(sub) else 0.0,
            "jss": float(sub["jss"].mean()) if len(sub) else 0.0,
        })

    return pd.DataFrame(out)


def collect_training_samples(base, train_dl, device, extra_schema, args):
    Xs, ys, ws, ps = [], [], [], []
    total = 0
    total_pos = 0

    base.eval()

    for batch in tqdm(train_dl, desc="collect R170 train samples"):
        batch = move_to_device(batch, device)

        with th.no_grad():
            res = base._common_step(batch, split="train", log=False)
            res = attach_raw_rich_features(base, batch, res, max_extra_dims=int(args.max_extra_dims))

            feats = candidate_features(
                res, batch,
                mz_max=float(base.hparams.mz_max),
                local_bin_res=float(args.local_bin_res),
                extra_schema=extra_schema,
            )

            args.mz_max = float(base.hparams.mz_max)
            residual, pos, target_mass, pred_prob = build_targets(res, args)

            sampled = sample_candidate_rows(
                feats=feats,
                residual=residual,
                pos=pos,
                target_mass=target_mass,
                pred_prob=pred_prob,
                batch=batch,
                results=res,
                args=args,
            )

        if sampled is None:
            continue

        X, y, w, p = sampled

        Xs.append(X)
        ys.append(y)
        ws.append(w)
        ps.append(p)

        total += X.shape[0]
        total_pos += int(p.sum())

        if total >= int(args.max_train_rows):
            break

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    p = np.concatenate(ps, axis=0)

    if X.shape[0] > int(args.max_train_rows):
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(X.shape[0], size=int(args.max_train_rows), replace=False)
        X, y, w, p = X[idx], y[idx], w[idx], p[idx]

    print("[R170] train sample shape:", X.shape)
    print("[R170] positive rate:", float(p.mean()))
    print("[R170] y mean/std/min/max:", float(y.mean()), float(y.std()), float(y.min()), float(y.max()))
    print("[R170] weight mean/std/min/max:", float(w.mean()), float(w.std()), float(w.min()), float(w.max()))

    return X, y, w, p


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--backend", default="auto", choices=["auto", "lightgbm", "lgbm", "sklearn"])

    ap.add_argument("--max_train_rows", type=int, default=1800000)
    ap.add_argument("--neg_topk_per_batch", type=int, default=1536)
    ap.add_argument("--neg_rand_per_batch", type=int, default=512)

    ap.add_argument("--mz_tol", type=float, default=0.01)
    ap.add_argument("--mz_sigma", type=float, default=0.003)
    ap.add_argument("--target_bin_res", type=float, default=0.01)
    ap.add_argument("--local_bin_res", type=float, default=0.01)
    ap.add_argument("--eval_bin_res", type=float, default=0.01)

    ap.add_argument("--residual_clip", type=float, default=6.0)
    ap.add_argument("--neg_residual", type=float, default=4.0)
    ap.add_argument("--score_clip", type=float, default=6.0)

    ap.add_argument("--low_w", type=float, default=0.50)
    ap.add_argument("--mid_w", type=float, default=2.00)
    ap.add_argument("--high_w", type=float, default=3.50)
    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--pos_intensity_weight", type=float, default=10.0)
    ap.add_argument("--neg_weight", type=float, default=1.0)
    ap.add_argument("--neg_prob_weight", type=float, default=20.0)

    # LightGBM params.
    ap.add_argument("--n_estimators", type=int, default=900)
    ap.add_argument("--gbdt_lr", type=float, default=0.035)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--min_child_samples", type=int, default=80)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample_bytree", type=float, default=0.85)
    ap.add_argument("--reg_alpha", type=float, default=0.1)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)

    # Sklearn fallback params.
    ap.add_argument("--hgb_iter", type=int, default=450)
    ap.add_argument("--hgb_lr", type=float, default=0.045)
    ap.add_argument("--hgb_leaf_nodes", type=int, default=63)
    ap.add_argument("--hgb_l2", type=float, default=0.02)

    ap.add_argument("--max_extra_dims", type=int, default=32)
    ap.add_argument("--alpha_grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,0.9,1.0,1.25,1.5")
    ap.add_argument("--load_regressor", default=None)
    ap.add_argument("--eval_test", action="store_true")

    args = ap.parse_args()

    np.random.seed(int(args.seed))
    th.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.template, args.config)
    cfg = force_r160_arch(cfg)

    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    base = FragGNNPL(**cfg)
    sd = load_state_dict_any(args.ckpt_path)
    missing, unexpected = base.load_state_dict(sd, strict=False)

    print("[R170] base missing:", len(missing))
    for x in missing[:20]:
        print("  missing:", x)
    print("[R170] base unexpected:", len(unexpected))
    for x in unexpected[:20]:
        print("  unexpected:", x)

    base = base.to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    if args.load_regressor is not None and str(args.load_regressor).lower() not in ["", "none", "null"]:
        with open(args.load_regressor, "rb") as f:
            pack = pickle.load(f)
        regressor = pack["model"]
        backend = pack.get("backend", "loaded")
        extra_schema = pack.get("extra_schema", [])
        print("[R170] loaded regressor:", args.load_regressor)
        print("[R170] loaded backend:", backend)
        print("[R170] loaded extra schema:", extra_schema)
    else:
        # Infer optional candidate-level extra features from one batch.
        first_batch = next(iter(train_dl))
        first_batch = move_to_device(first_batch, device)
        with th.no_grad():
            first_res = base._common_step(first_batch, split="train", log=False)
            first_res = attach_raw_rich_features(base, first_batch, first_res, max_extra_dims=int(args.max_extra_dims))
        extra_schema = infer_extra_schema(first_res, max_extra_dims=int(args.max_extra_dims))

        print("[R170] inferred extra schema:", extra_schema)
        with open(out_dir / "r170_extra_schema.pkl", "wb") as f:
            pickle.dump(extra_schema, f)

        X, y, w, p = collect_training_samples(base, train_dl, device, extra_schema, args)

        np.savez_compressed(
            out_dir / "r170_train_sample_stats.npz",
            y=y,
            w=w,
            p=p,
        )

        regressor, backend = train_regressor(X, y, w, args)

        with open(out_dir / "r170_regressor.pkl", "wb") as f:
            pickle.dump({
                "model": regressor,
                "backend": backend,
                "extra_schema": extra_schema,
                "args": vars(args),
            }, f)

    alpha_rows = []
    alpha_values = [float(x) for x in args.alpha_grid.split(",")]

    for alpha in alpha_values:
        tab = eval_split(base, regressor, extra_schema, val_dl, device, args, split="val", alpha=alpha)
        g = tab[tab["ce_bucket"] == "global"].iloc[0]
        alpha_rows.append({
            "alpha": alpha,
            "val_cos": float(g["cos"]),
            "val_jss": float(g["jss"]),
        })
        tab.to_csv(out_dir / f"r170_val_alpha{alpha}.csv", index=False)

    adf = pd.DataFrame(alpha_rows)
    adf.to_csv(out_dir / "r170_alpha_val.csv", index=False)
    print("[R170 alpha val]")
    print(adf.to_string(index=False))

    best = adf.sort_values("val_cos", ascending=False).iloc[0]
    best_alpha = float(best["alpha"])

    print("[R170] best alpha:", best_alpha, "best val cos:", float(best["val_cos"]))

    best_val = eval_split(base, regressor, extra_schema, val_dl, device, args, split="val", alpha=best_alpha)
    best_val.to_csv(out_dir / "r170_best_val.csv", index=False)
    print("\n===== R170 BEST VAL =====")
    print(best_val.to_string(index=False))

    if args.eval_test:
        best_test = eval_split(base, regressor, extra_schema, test_dl, device, args, split="test", alpha=best_alpha)
        best_test.to_csv(out_dir / "r170_best_test.csv", index=False)
        print("\n===== R170 BEST TEST =====")
        print(best_test.to_string(index=False))

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
