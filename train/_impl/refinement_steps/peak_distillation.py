import argparse
import math
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


def find_ce(batch):
    keys = [
        "spec_ce", "spec_ces",
        "true_ces", "ce", "ces",
        "collision_energy", "collision_energies",
        "ace", "aces",
    ]
    for k in keys:
        if isinstance(batch, dict) and k in batch and isinstance(batch[k], th.Tensor):
            v = batch[k].reshape(-1).float()
            if v.numel() > 0:
                return v, k

    for k, v in batch.items():
        if not isinstance(v, th.Tensor):
            continue
        lk = str(k).lower()
        if ("ce" in lk or "collision" in lk or "energy" in lk) and v.numel() > 0:
            return v.reshape(-1).float(), k

    raise RuntimeError("cannot find CE tensor in batch")


def ce_bucket_names(ce):
    names = []
    for x in ce.detach().cpu().numpy().reshape(-1):
        x = float(x)
        if x <= 20:
            names.append("low_<=20")
        elif x <= 40:
            names.append("mid_20_40")
        else:
            names.append("high_>40")
    return names


def ce_weights(ce, low_w, mid_w, high_w):
    w = th.ones_like(ce, dtype=th.float)
    w = th.where(ce <= 20, th.full_like(w, float(low_w)), w)
    w = th.where((ce > 20) & (ce <= 40), th.full_like(w, float(mid_w)), w)
    w = th.where(ce > 40, th.full_like(w, float(high_w)), w)
    return w


def dense_by_round_bins(mzs, vals, batch_idxs, batch_size, mz_max, bin_res, binary=False):
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
    if binary:
        dense_flat.index_add_(0, flat, th.ones_like(vals))
        dense = dense_flat.reshape(batch_size, num_bins)
        return (dense > 0).to(dtype)

    dense_flat.index_add_(0, flat, vals)
    return dense_flat.reshape(batch_size, num_bins)


def cosine_dense(a, b, eps=1e-12):
    num = (a * b).sum(dim=1)
    den = (a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt()).clamp_min(eps)
    return num / den


def jss_dense(a, b, eps=1e-12):
    a = a.clamp_min(0)
    b = b.clamp_min(0)
    a = a / a.sum(dim=1, keepdim=True).clamp_min(eps)
    b = b / b.sum(dim=1, keepdim=True).clamp_min(eps)
    m = 0.5 * (a + b)

    kl_am = (a * ((a + eps).log() - (m + eps).log())).sum(dim=1)
    kl_bm = (b * ((b + eps).log() - (m + eps).log())).sum(dim=1)
    jsd = 0.5 * (kl_am + kl_bm)

    return 1.0 - jsd / math.log(2.0)


def override_cfg(cfg, args):
    # K3b formula composition residual.
    cfg["use_formula_comp_residual"] = True
    cfg["formula_comp_hidden_size"] = int(args.k3b_hidden)
    cfg["formula_comp_dropout"] = float(args.k3b_dropout)
    cfg["formula_comp_delta_scale"] = float(args.k3b_delta_scale)
    cfg["formula_comp_center_per_spectrum"] = True
    cfg["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    cfg.setdefault("frag_params", {})
    cfg["frag_params"]["formula_comp_feats"] = True
    cfg["frag_params"]["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    # CE-response scorer.
    cfg["use_ce_response_scorer"] = True
    cfg["ce_response_hidden_size"] = int(args.ce_hidden)
    cfg["ce_response_dropout"] = float(args.ce_dropout)
    cfg["ce_response_delta_scale"] = float(args.ce_delta_scale)
    cfg["ce_response_use_formula_comp"] = bool(args.ce_use_formula_comp)
    cfg["ce_response_use_depth"] = bool(args.ce_use_depth)
    cfg["ce_response_use_h"] = bool(args.ce_use_h)

    # CEFlowFrag.
    cfg["use_ce_flowfrag"] = bool(args.use_ce_flowfrag)
    cfg["ce_flowfrag_lambda_max"] = float(args.ce_flowfrag_lambda_max)
    cfg["ce_flowfrag_hidden_size"] = int(args.ce_flowfrag_hidden)
    cfg["ce_flowfrag_dropout"] = float(args.ce_flowfrag_dropout)
    cfg["ce_flowfrag_max_depth"] = int(args.ce_flowfrag_max_depth)
    cfg["ce_flowfrag_mixture_hidden_size"] = int(args.ce_flowfrag_mixture_hidden)
    cfg["ce_flowfrag_mixture_dropout"] = float(args.ce_flowfrag_mixture_dropout)
    cfg["ce_flowfrag_mixture_init_bias"] = float(args.ce_flowfrag_mixture_init_bias)
    cfg["ce_flowfrag_delta_clip"] = float(args.ce_flowfrag_delta_clip)
    cfg["ce_flowfrag_use_direct_node"] = bool(args.ce_flowfrag_use_direct_node)
    cfg["ce_flowfrag_direct_mix"] = float(args.ce_flowfrag_direct_mix)

    # Real official-path binned renderer.
    cfg["use_binned_spectrum_renderer"] = True
    cfg["binned_spectrum_renderer_apply_train"] = True
    cfg["binned_spectrum_renderer_bin_res"] = float(args.bin_res)
    cfg["binned_spectrum_renderer_preserve_mass"] = True
    cfg["binned_spectrum_renderer_max_bins"] = int(args.max_bins)

    # CE-weighted binned aux loss.
    cfg["use_ce_weighted_binned_aux_loss"] = float(args.ce_binned_aux_weight) > 0
    cfg["ce_binned_aux_loss_weight"] = float(args.ce_binned_aux_weight)
    cfg["ce_binned_aux_mid_threshold"] = 20.0
    cfg["ce_binned_aux_high_threshold"] = 40.0
    cfg["ce_binned_aux_low_weight"] = float(args.low_w)
    cfg["ce_binned_aux_mid_weight"] = float(args.mid_w)
    cfg["ce_binned_aux_high_weight"] = float(args.high_w)

    # Keep old false-mass aux off in R160. We do our own candidate-level false loss.
    cfg["use_r117_false_mass_aux_loss"] = False
    cfg["r117_weight"] = 0.0

    return cfg


def allowed_prefixes(args):
    allow = [
        "formula_comp_residual_head",
        "ce_response_scorer",
        "formula_module",
    ]

    if args.train_k3b:
        allow.append("formula_comp_residual_head")

    if args.train_formula_module:
        allow.append("formula_module")

    if args.train_frag_rep:
        allow.extend([
            "frag_embedder",
            "frag_pool",
            "formula_embedder",
            "depth_embedder",
            "complement_embedder",
            "cc_interstage",
        ])

    if args.train_mol_rep:
        allow.extend([
            "mol_embedder",
            "mol_pool",
        ])

    if args.train_ce_flowfrag:
        allow.extend([
            "ce_flowfrag_edge_head",
            "ce_flowfrag_direct_head",
            "ce_flowfrag_mixture_head",
        ])

    if args.train_refiner:
        allow.append("spectrum_candidate_refiner")

    if args.train_render_gate:
        allow.append("rendered_peak_drop_gate")

    # remove duplicates, keep order
    out = []
    for x in allow:
        if x not in out:
            out.append(x)
    return out


def freeze_model(model, args):
    allow = allowed_prefixes(args)

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        short = name[6:] if name.startswith("model.") else name
        if any(short.startswith(prefix) for prefix in allow):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"[R160] trainable params: {trainable} / {total}")
    print(f"[R160] allowed prefixes: {allow}")

    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  trainable:", name, tuple(p.shape))
            shown += 1
            if shown >= 50:
                print("  ...")
                break

    return allow


def set_partial_train_mode(model, allow):
    model.eval()
    for name, module in model.model.named_modules():
        if any(name.startswith(prefix) for prefix in allow):
            module.train()


def build_peak_oracle_loss(results, batch, model, args):
    pred_mzs = results["pred_mzs"]
    pred_logp = results["pred_logprobs"]
    pred_prob = pred_logp.exp()
    pred_bidx = results["pred_batch_idxs"].long()

    true_mzs = results["true_mzs"]
    true_prob = results["true_logprobs"].exp()
    true_bidx = results["true_batch_idxs"].long()

    batch_size = int(results["unique_id"].numel())
    mz_max = float(model.hparams.mz_max)
    bin_res = float(args.oracle_bin_res)
    mz_tol = float(args.oracle_mz_tol)
    sigma = float(args.oracle_mz_sigma)

    true_dense = dense_by_round_bins(
        true_mzs, true_prob, true_bidx,
        batch_size=batch_size,
        mz_max=mz_max,
        bin_res=bin_res,
        binary=False,
    )

    num_bins = true_dense.shape[1]
    pred_bins0 = th.round(pred_mzs / bin_res).long().clamp(0, num_bins - 1)

    k = max(0, int(math.ceil(mz_tol / bin_res)))

    best_score = th.zeros_like(pred_prob)
    best_mass = th.zeros_like(pred_prob)
    best_weight = th.zeros_like(pred_prob)
    best_bin = pred_bins0.clone()

    # Match each predicted candidate entry to the nearest/highest true peak bin within tolerance.
    for off in range(-k, k + 1):
        idx = (pred_bins0 + off).clamp(0, num_bins - 1)
        center_mz = idx.to(pred_mzs.dtype) * bin_res
        dist = (pred_mzs - center_mz).abs()
        within = dist <= mz_tol

        true_mass = true_dense[pred_bidx, idx]
        gauss = th.exp(-0.5 * (dist / max(sigma, 1e-12)).pow(2))
        score = true_mass * gauss * within.to(true_mass.dtype)

        improve = score > best_score
        best_score = th.where(improve, score, best_score)
        best_mass = th.where(improve, true_mass, best_mass)
        best_weight = th.where(improve, gauss, best_weight)
        best_bin = th.where(improve, idx, best_bin)

    valid = best_score > 0

    # Distribute one true peak's mass across all candidate entries that match that same true bin.
    target = th.zeros_like(pred_prob)

    if valid.any():
        flat_group = pred_bidx[valid] * num_bins + best_bin[valid]
        denom = th.zeros(batch_size * num_bins, device=pred_prob.device, dtype=pred_prob.dtype)
        denom.index_add_(0, flat_group, best_weight[valid].clamp_min(1e-12))

        target_valid = best_mass[valid] * best_weight[valid] / denom[flat_group].clamp_min(1e-12)
        target[valid] = target_valid

    ce, ce_key = find_ce(batch)
    ce = ce.to(pred_prob.device).reshape(-1).float()
    w = ce_weights(ce, args.oracle_low_w, args.oracle_mid_w, args.oracle_high_w).to(pred_prob.device)
    w = w / w.mean().clamp_min(1e-12)

    ce_num = th.zeros(batch_size, device=pred_prob.device, dtype=pred_prob.dtype)
    ce_num.index_add_(0, pred_bidx, -(target * pred_logp))

    target_sum = th.zeros(batch_size, device=pred_prob.device, dtype=pred_prob.dtype)
    target_sum.index_add_(0, pred_bidx, target)

    peak_ce_per_spec = ce_num / target_sum.clamp_min(1e-12)

    false_mask = ~valid
    false_mass = th.zeros(batch_size, device=pred_prob.device, dtype=pred_prob.dtype)
    if false_mask.any():
        false_mass.index_add_(0, pred_bidx[false_mask], pred_prob[false_mask])

    hit_pred_mass = th.zeros(batch_size, device=pred_prob.device, dtype=pred_prob.dtype)
    if valid.any():
        hit_pred_mass.index_add_(0, pred_bidx[valid], pred_prob[valid])

    # Main oracle terms.
    peak_ce_loss = (peak_ce_per_spec * w).sum() / w.sum().clamp_min(1e-12)
    false_mass_loss = (false_mass * w).sum() / w.sum().clamp_min(1e-12)
    hit_mass_loss = (-(hit_pred_mass.clamp_min(1e-8)).log() * w).sum() / w.sum().clamp_min(1e-12)

    return {
        "peak_ce_loss": peak_ce_loss,
        "false_mass_loss": false_mass_loss,
        "hit_mass_loss": hit_mass_loss,
        "target_hit_mass": target_sum.mean().detach(),
        "pred_hit_mass": hit_pred_mass.mean().detach(),
        "pred_false_mass": false_mass.mean().detach(),
        "valid_frac": valid.float().mean().detach(),
        "ce_key": ce_key,
    }


def build_r180_spectrum_loss(results, batch, model, args):
    """
    R180B: train-time spectrum-level CE-weighted cosine/JSS loss.

    This is intentionally inside the R160 training script so the model architecture
    stays exactly aligned with R160A/R172E.
    """
    pred_mzs = results["pred_mzs"]
    pred_logp = results["pred_logprobs"]
    pred_prob = pred_logp.exp()
    pred_bidx = results["pred_batch_idxs"].long()

    true_mzs = results["true_mzs"]
    true_prob = results["true_logprobs"].exp()
    true_bidx = results["true_batch_idxs"].long()

    batch_size = int(results["unique_id"].numel())
    mz_max = float(model.hparams.mz_max)
    bin_res = float(args.spectrum_bin_res)

    true_dense = dense_by_round_bins(
        true_mzs,
        true_prob,
        true_bidx,
        batch_size=batch_size,
        mz_max=mz_max,
        bin_res=bin_res,
        binary=False,
    )

    pred_dense = dense_by_round_bins(
        pred_mzs,
        pred_prob,
        pred_bidx,
        batch_size=batch_size,
        mz_max=mz_max,
        bin_res=bin_res,
        binary=False,
    )

    cos = cosine_dense(true_dense, pred_dense).clamp(0.0, 1.0)
    jss = jss_dense(true_dense, pred_dense).clamp(0.0, 1.0)

    ce, ce_key = find_ce(batch)
    ce = ce.to(pred_prob.device).reshape(-1).float()

    w = ce_weights(
        ce,
        args.spectrum_low_w,
        args.spectrum_mid_w,
        args.spectrum_high_w,
    ).to(pred_prob.device)
    w = w / w.mean().clamp_min(1e-12)

    cos_dist = 1.0 - cos
    jss_dist = 1.0 - jss

    spec_loss_vec = (
        float(args.spectrum_cos_weight) * cos_dist
        + float(args.spectrum_jss_weight) * jss_dist
    )

    spec_loss = (spec_loss_vec * w).sum() / w.sum().clamp_min(1e-12)

    return {
        "spectrum_loss": spec_loss,
        "spectrum_cos": cos.mean().detach(),
        "spectrum_jss": jss.mean().detach(),
        "spectrum_cos_dist": cos_dist.mean().detach(),
        "spectrum_jss_dist": jss_dist.mean().detach(),
        "spectrum_ce_weight": w.mean().detach(),
        "ce_key": ce_key,
    }


@th.no_grad()
def eval_split(model, dl, device, args, split):
    model.eval()
    rows = []

    for batch in tqdm(dl, desc=f"eval {split}"):
        batch = move_to_device(batch, device)
        results = model._common_step(batch, split=split, log=False)

        unique_ids = results["unique_id"].detach().cpu().reshape(-1).numpy().astype(int)
        batch_size = len(unique_ids)

        true_dense = dense_by_round_bins(
            results["true_mzs"],
            results["true_logprobs"].exp(),
            results["true_batch_idxs"],
            batch_size=batch_size,
            mz_max=float(model.hparams.mz_max),
            bin_res=float(args.eval_bin_res),
            binary=False,
        )

        pred_dense = dense_by_round_bins(
            results["pred_mzs"],
            results["pred_logprobs"].exp(),
            results["pred_batch_idxs"],
            batch_size=batch_size,
            mz_max=float(model.hparams.mz_max),
            bin_res=float(args.eval_bin_res),
            binary=False,
        )

        cos = cosine_dense(true_dense, pred_dense)
        jss = jss_dense(true_dense, pred_dense)

        ce, _ = find_ce(batch)
        ce = ce.detach().cpu().reshape(-1).numpy().astype(float)
        buckets = ce_bucket_names(th.tensor(ce))

        for i, sid in enumerate(unique_ids):
            rows.append({
                "spec_id": int(sid),
                "ce": float(ce[i]),
                "ce_bucket": buckets[i],
                "cos": float(cos[i].detach().cpu()),
                "jss": float(jss[i].detach().cpu()),
            })

    df = pd.DataFrame(rows)
    global_row = {
        "ce_bucket": "global",
        "spec_count": int(len(df)),
        "mean_ce": float(df["ce"].mean()),
        "cos": float(df["cos"].mean()),
        "jss": float(df["jss"].mean()),
    }

    bucket_rows = []
    for bucket in ["low_<=20", "mid_20_40", "high_>40"]:
        sub = df[df["ce_bucket"] == bucket]
        if len(sub) == 0:
            continue
        bucket_rows.append({
            "ce_bucket": bucket,
            "spec_count": int(len(sub)),
            "mean_ce": float(sub["ce"].mean()),
            "cos": float(sub["cos"].mean()),
            "jss": float(sub["jss"].mean()),
        })

    out = pd.DataFrame([global_row] + bucket_rows)
    return out, df


def print_eval(tag, table):
    g = table[table["ce_bucket"] == "global"].iloc[0]
    print(f"\n===== {tag} =====")
    print(f"global cos: {g['cos']}")
    print(f"global jss: {g['jss']}")
    print(table.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max_train_batches", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--k3b_hidden", type=int, default=128)
    ap.add_argument("--k3b_dropout", type=float, default=0.05)
    ap.add_argument("--k3b_delta_scale", type=float, default=0.05)
    ap.add_argument("--formula_comp_feat_size", type=int, default=18)

    ap.add_argument("--ce_hidden", type=int, default=128)
    ap.add_argument("--ce_dropout", type=float, default=0.05)
    ap.add_argument("--ce_delta_scale", type=float, default=0.020)
    ap.add_argument("--ce_use_formula_comp", action="store_true")
    ap.add_argument("--ce_use_depth", action="store_true")
    ap.add_argument("--ce_use_h", action="store_true")

    ap.add_argument("--use_ce_flowfrag", action="store_true")
    ap.add_argument("--ce_flowfrag_lambda_max", type=float, default=0.15)
    ap.add_argument("--ce_flowfrag_hidden", type=int, default=128)
    ap.add_argument("--ce_flowfrag_dropout", type=float, default=0.05)
    ap.add_argument("--ce_flowfrag_max_depth", type=int, default=4)
    ap.add_argument("--ce_flowfrag_mixture_hidden", type=int, default=128)
    ap.add_argument("--ce_flowfrag_mixture_dropout", type=float, default=0.05)
    ap.add_argument("--ce_flowfrag_mixture_init_bias", type=float, default=-3.0)
    ap.add_argument("--ce_flowfrag_delta_clip", type=float, default=3.0)
    ap.add_argument("--ce_flowfrag_use_direct_node", action="store_true")
    ap.add_argument("--ce_flowfrag_direct_mix", type=float, default=0.35)

    ap.add_argument("--bin_res", type=float, default=0.01)
    ap.add_argument("--max_bins", type=int, default=0)
    ap.add_argument("--eval_bin_res", type=float, default=0.01)

    ap.add_argument("--ce_binned_aux_weight", type=float, default=0.0015)
    ap.add_argument("--low_w", type=float, default=0.25)
    ap.add_argument("--mid_w", type=float, default=1.75)
    ap.add_argument("--high_w", type=float, default=2.25)

    # R160 oracle losses.
    ap.add_argument("--peak_oracle_weight", type=float, default=0.02)
    ap.add_argument("--false_mass_weight", type=float, default=0.015)
    ap.add_argument("--hit_mass_weight", type=float, default=0.003)
    ap.add_argument("--oracle_bin_res", type=float, default=0.01)
    ap.add_argument("--oracle_mz_tol", type=float, default=0.01)
    ap.add_argument("--oracle_mz_sigma", type=float, default=0.003)
    ap.add_argument("--oracle_low_w", type=float, default=0.50)
    ap.add_argument("--oracle_mid_w", type=float, default=2.00)
    ap.add_argument("--oracle_high_w", type=float, default=3.00)

    # R180B spectrum-level CE-weighted loss.
    ap.add_argument("--spectrum_loss_weight", type=float, default=0.0)
    ap.add_argument("--spectrum_cos_weight", type=float, default=1.0)
    ap.add_argument("--spectrum_jss_weight", type=float, default=0.25)
    ap.add_argument("--spectrum_bin_res", type=float, default=0.01)
    ap.add_argument("--spectrum_low_w", type=float, default=0.50)
    ap.add_argument("--spectrum_mid_w", type=float, default=2.50)
    ap.add_argument("--spectrum_high_w", type=float, default=5.00)

    # Train scopes.
    ap.add_argument("--train_k3b", action="store_true")
    ap.add_argument("--train_formula_module", action="store_true")
    ap.add_argument("--train_frag_rep", action="store_true")
    ap.add_argument("--train_mol_rep", action="store_true")
    ap.add_argument("--train_ce_flowfrag", action="store_true")
    ap.add_argument("--train_refiner", action="store_true")
    ap.add_argument("--train_render_gate", action="store_true")

    ap.add_argument("--eval_test", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.template, args.config)
    cfg = override_cfg(cfg, args)

    print("loading datasets")
    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    model = FragGNNPL(**cfg)
    sd = load_state_dict_any(args.ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[R160] missing keys:", len(missing))
    for x in missing[:30]:
        print("  missing:", x)
    print("[R160] unexpected keys:", len(unexpected))
    for x in unexpected[:30]:
        print("  unexpected:", x)

    model = model.to(device)
    allow = freeze_model(model, args)

    opt = th.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    before_table, _ = eval_split(model, val_dl, device, args, split="val")
    print_eval("R160 BEFORE VAL", before_table)
    before_table.to_csv(out_dir / "r160_val_epoch0_before.csv", index=False)

    best_score = -1e18
    best_epoch = -1
    train_rows = []

    for epoch in range(1, args.epochs + 1):
        set_partial_train_mode(model, allow)

        sums = {
            "train_loss": 0.0,
            "base_loss": 0.0,
            "peak_ce_loss": 0.0,
            "false_mass_loss": 0.0,
            "hit_mass_loss": 0.0,
            "target_hit_mass": 0.0,
            "pred_hit_mass": 0.0,
            "pred_false_mass": 0.0,
            "valid_frac": 0.0,
            "r180_spectrum_loss": 0.0,
            "r180_spectrum_cos": 0.0,
            "r180_spectrum_jss": 0.0,
            "r180_spectrum_ce_weight": 0.0,
        }
        n_steps = 0

        pbar = tqdm(train_dl, desc=f"R160 train epoch={epoch}")
        for step, batch in enumerate(pbar):
            if args.max_train_batches > 0 and step >= args.max_train_batches:
                break

            batch = move_to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            set_partial_train_mode(model, allow)
            res = model._common_step(batch, split="train", log=False)
            base_loss = res["mean_loss"]

            oracle = build_peak_oracle_loss(res, batch, model, args)

            loss = (
                base_loss
                + float(args.peak_oracle_weight) * oracle["peak_ce_loss"]
                + float(args.false_mass_weight) * oracle["false_mass_loss"]
                + float(args.hit_mass_weight) * oracle["hit_mass_loss"]
            )

            r180 = None
            if float(args.spectrum_loss_weight) > 0:
                r180 = build_r180_spectrum_loss(res, batch, model, args)
                loss = loss + float(args.spectrum_loss_weight) * r180["spectrum_loss"]

            loss.backward()
            th.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

            n_steps += 1
            sums["train_loss"] += float(loss.detach().cpu())
            sums["base_loss"] += float(base_loss.detach().cpu())
            for k in [
                "peak_ce_loss", "false_mass_loss", "hit_mass_loss",
                "target_hit_mass", "pred_hit_mass", "pred_false_mass", "valid_frac",
            ]:
                sums[k] += float(oracle[k].detach().cpu())

            if r180 is not None:
                sums["r180_spectrum_loss"] += float(r180["spectrum_loss"].detach().cpu())
                sums["r180_spectrum_cos"] += float(r180["spectrum_cos"].detach().cpu())
                sums["r180_spectrum_jss"] += float(r180["spectrum_jss"].detach().cpu())
                sums["r180_spectrum_ce_weight"] += float(r180["spectrum_ce_weight"].detach().cpu())

            postfix = {
                "loss": f"{float(loss.detach().cpu()):.4f}",
                "base": f"{float(base_loss.detach().cpu()):.4f}",
                "peakCE": f"{float(oracle['peak_ce_loss'].detach().cpu()):.3f}",
                "false": f"{float(oracle['false_mass_loss'].detach().cpu()):.3f}",
                "hitP": f"{float(oracle['pred_hit_mass'].detach().cpu()):.3f}",
            }
            if r180 is not None:
                postfix["spLoss"] = f"{float(r180['spectrum_loss'].detach().cpu()):.3f}"
                postfix["spCos"] = f"{float(r180['spectrum_cos'].detach().cpu()):.3f}"
            pbar.set_postfix(postfix)

        row = {"epoch": epoch, "n_steps": n_steps}
        for k, v in sums.items():
            row[k] = v / max(n_steps, 1)
        train_rows.append(row)
        pd.DataFrame(train_rows).to_csv(out_dir / "r160_train_log.csv", index=False)
        print("[R160 train]", row)

        val_table, _ = eval_split(model, val_dl, device, args, split="val")
        g = val_table[val_table["ce_bucket"] == "global"].iloc[0]
        # Stronger than old score: still keeps JSS in selection.
        score = float(g["cos"])

        print_eval(f"R160 VAL epoch={epoch} score={score:.6f}", val_table)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            th.save(model.state_dict(), out_dir / "r160_best_state.pt")
            val_table.to_csv(out_dir / "r160_best_val.csv", index=False)
            print("[R160] saved best:", out_dir / "r160_best_state.pt")

    print("[R160] loaded best epoch:", best_epoch, "score:", best_score)
    best_sd = th.load(out_dir / "r160_best_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_sd, strict=False)
    model = model.to(device)
    model.eval()

    best_val, _ = eval_split(model, val_dl, device, args, split="val")
    print_eval("R160 BEST VAL", best_val)
    best_val.to_csv(out_dir / "r160_best_val.csv", index=False)

    if args.eval_test:
        best_test, _ = eval_split(model, test_dl, device, args, split="test")
        print_eval("R160 BEST TEST", best_test)
        best_test.to_csv(out_dir / "r160_best_test.csv", index=False)

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
