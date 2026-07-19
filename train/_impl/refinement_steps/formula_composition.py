import argparse
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
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if hasattr(obj, "to") and not isinstance(obj, (str, bytes)):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def load_state_dict(path):
    ckpt = th.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def override_cfg(cfg, args):
    # ===== 正确位置：启用模型内部 K3b formula composition residual =====
    cfg["use_formula_comp_residual"] = True
    cfg["formula_comp_hidden_size"] = int(args.hidden)
    cfg["formula_comp_dropout"] = float(args.dropout)
    cfg["formula_comp_delta_scale"] = float(args.delta_scale)
    cfg["formula_comp_center_per_spectrum"] = True
    cfg["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    # ===== 关键修复：dataset 侧也必须生成 frag_formula_comp_feats =====
    # model.py 里 K3b 会 assert batch 中存在 frag_formula_comp_feats；
    # 这个字段只由 dataset 在 frag_params.formula_comp_feats=True 时生成。
    cfg.setdefault("frag_params", {})
    cfg["frag_params"]["formula_comp_feats"] = True
    cfg["frag_params"]["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    # ===== 训练也走 R98 0.01Da official-style renderer，而不是 train/eval mismatch =====
    cfg["use_binned_spectrum_renderer"] = True
    cfg["binned_spectrum_renderer_apply_train"] = True
    cfg["binned_spectrum_renderer_bin_res"] = float(args.bin_res)
    cfg["binned_spectrum_renderer_preserve_mass"] = True
    cfg["binned_spectrum_renderer_max_bins"] = int(args.max_bins)

    # ===== 用已有 CE-weighted binned aux，直接对官方 bin intensity 对齐 =====
    cfg["use_ce_weighted_binned_aux_loss"] = bool(args.ce_binned_aux_weight > 0)
    cfg["ce_binned_aux_loss_weight"] = float(args.ce_binned_aux_weight)
    cfg["ce_binned_aux_mid_threshold"] = 20.0
    cfg["ce_binned_aux_high_threshold"] = 40.0
    cfg["ce_binned_aux_low_weight"] = float(args.low_w)
    cfg["ce_binned_aux_mid_weight"] = float(args.mid_w)
    cfg["ce_binned_aux_high_weight"] = float(args.high_w)

    # ===== R117 是 fixed-support intensity allocation，先允许小权重 =====
    cfg["use_r117_support_oracle_reweight_loss"] = bool(args.r117_weight > 0)
    cfg["r117_support_oracle_weight"] = float(args.r117_weight)
    cfg["r117_oracle_bin_res"] = float(args.bin_res)
    cfg["r117_false_mass_weight"] = float(args.r117_false_weight)
    cfg["r117_min_covered_true_mass"] = 1.0e-8
    cfg["r117_eps"] = 1.0e-12

    return cfg


def freeze_for_k3b(model, train_formula_module=False, train_refiner=False):
    for p in model.parameters():
        p.requires_grad_(False)

    allow = ["formula_comp_residual_head"]
    if train_formula_module:
        allow.append("formula_module")
    if train_refiner:
        allow.append("spectrum_candidate_refiner")

    names = []
    for name, p in model.model.named_parameters():
        if any(name.startswith(prefix) for prefix in allow):
            p.requires_grad_(True)
            names.append(name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print("[R146] trainable params:", n_train, "/", n_total)
    print("[R146] trainable prefixes:", allow)
    print("[R146] first trainable names:")
    for n in names[:30]:
        print("  ", n)
    if len(names) > 30:
        print("  ...")

    if n_train == 0:
        raise RuntimeError("No trainable params. K3b formula_comp_residual_head not found.")


def set_k3b_train_mode(model, train_formula_module=False, train_refiner=False):
    # 冻结主干时，主模型保持 eval，避免 frozen dropout 改分布。
    model.eval()

    for name, module in model.model.named_modules():
        if name.startswith("formula_comp_residual_head"):
            module.train()
        if train_formula_module and name.startswith("formula_module"):
            module.train()
        if train_refiner and name.startswith("spectrum_candidate_refiner"):
            module.train()


def pick_metric_key(res, token):
    keys = []
    for k, v in res.items():
        if token in k and isinstance(v, th.Tensor):
            if token == "recall" and "wrecall" in k:
                continue
            if token == "precision" and "wprecision" in k:
                continue
            keys.append(k)
    if not keys:
        return None

    # 优先官方 0.01 binned cos/jss
    def score_key(k):
        s = 0
        if "0.01" in k:
            s -= 10
        if "sqrt" in k:
            s += 5
        if "opt" in k:
            s += 5
        return (s, len(k), k)

    return sorted(keys, key=score_key)[0]


def get_vec(res, key, bs):
    if key is None or key not in res:
        return np.full(bs, np.nan, dtype=np.float64)
    x = res[key]
    if isinstance(x, th.Tensor):
        x = x.detach().float().reshape(-1).cpu().numpy()
    else:
        x = np.asarray(x).reshape(-1)
    if len(x) != bs:
        return np.resize(x, bs)
    return x


def ce_bucket(x):
    x = float(x)
    if x <= 20:
        return "low_<=20"
    if x <= 40:
        return "mid_20_40"
    return "high_>40"


def get_ce_np(res, batch, bs):
    ce = res.get("input_ce", None)
    if ce is None:
        ce = batch.get("spec_ce", None)
    if ce is None:
        return np.zeros(bs, dtype=np.float32)
    if isinstance(ce, th.Tensor):
        ce = ce.detach().float().reshape(-1).cpu().numpy()
    else:
        ce = np.asarray(ce).reshape(-1)
    if len(ce) != bs:
        ce = np.resize(ce, bs)
    return ce.astype(np.float32)


def eval_buckets(model, dl, device, split):
    model.eval()
    rows = []

    with th.inference_mode():
        for batch in tqdm(dl, desc=f"eval {split}"):
            batch = move_to_device(batch, device)
            res = model._common_step(batch, split=split, log=False)

            bs = int(batch["batch_size"].detach().cpu().item()) if hasattr(batch["batch_size"], "detach") else int(batch["batch_size"])
            ce_np = get_ce_np(res, batch, bs)

            cos_key = pick_metric_key(res, "cos_sim")
            jss_key = pick_metric_key(res, "jss")
            prec_key = pick_metric_key(res, "precision")
            recall_key = pick_metric_key(res, "recall")
            wrecall_key = pick_metric_key(res, "wrecall")

            cos = get_vec(res, cos_key, bs)
            jss = get_vec(res, jss_key, bs)
            prec = get_vec(res, prec_key, bs)
            recall = get_vec(res, recall_key, bs)
            wrecall = get_vec(res, wrecall_key, bs)

            for i in range(bs):
                rows.append({
                    "split": split,
                    "ce": float(ce_np[i]),
                    "ce_bucket": ce_bucket(ce_np[i]),
                    "cos": float(cos[i]),
                    "jss": float(jss[i]),
                    "precision": float(prec[i]),
                    "recall": float(recall[i]),
                    "wrecall": float(wrecall[i]),
                })

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("ce_bucket")
        .agg(
            spec_count=("ce", "count"),
            mean_ce=("ce", "mean"),
            cos=("cos", "mean"),
            jss=("jss", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            wrecall=("wrecall", "mean"),
        )
        .reset_index()
    )

    order = {"low_<=20": 0, "mid_20_40": 1, "high_>40": 2}
    summary["order"] = summary["ce_bucket"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"])
    return summary


def global_metric(summary, col="cos"):
    return float((summary[col] * summary["spec_count"]).sum() / summary["spec_count"].sum())


def score_val(summary, baseline):
    g = global_metric(summary, "cos")

    def get(bucket, col):
        s = summary[summary["ce_bucket"] == bucket]
        return float(s[col].iloc[0]) if len(s) else np.nan

    def getb(bucket, col):
        s = baseline[baseline["ce_bucket"] == bucket]
        return float(s[col].iloc[0]) if len(s) else np.nan

    low = get("low_<=20", "cos")
    mid = get("mid_20_40", "cos")
    high = get("high_>40", "cos")

    low0 = getb("low_<=20", "cos")
    mid0 = getb("mid_20_40", "cos")
    jss0 = global_metric(baseline, "jss")
    jss = global_metric(summary, "jss")

    low_drop = max(0.0, low0 - low)
    mid_drop = max(0.0, mid0 - mid)
    jss_drop = max(0.0, jss0 - jss)

    # 不允许像 R145 那样 cos 微涨但 JSS 大掉。
    return g + 0.50 * mid + 0.05 * high - 5.0 * low_drop - 4.0 * mid_drop - 3.0 * jss_drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_train_batches", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--delta_scale", type=float, default=0.05)
    ap.add_argument("--formula_comp_feat_size", type=int, default=18)

    ap.add_argument("--bin_res", type=float, default=0.01)
    ap.add_argument("--max_bins", type=int, default=0)

    ap.add_argument("--ce_binned_aux_weight", type=float, default=0.0015)
    ap.add_argument("--r117_weight", type=float, default=0.0)
    ap.add_argument("--r117_false_weight", type=float, default=0.25)

    ap.add_argument("--low_w", type=float, default=0.30)
    ap.add_argument("--mid_w", type=float, default=1.50)
    ap.add_argument("--high_w", type=float, default=2.00)

    ap.add_argument("--train_formula_module", action="store_true")
    ap.add_argument("--train_refiner", action="store_true")
    ap.add_argument("--eval_test", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(args)]).to_csv(out_dir / "r146_args.csv", index=False)

    cfg = load_config(args.template, args.config)
    cfg = override_cfg(cfg, args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    model = FragGNNPL(**cfg)
    sd = load_state_dict(args.ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[R146] missing keys:", len(missing))
    for k in missing[:20]:
        print("  missing:", k)
    print("[R146] unexpected keys:", len(unexpected))

    model = model.to(device)

    freeze_for_k3b(
        model,
        train_formula_module=args.train_formula_module,
        train_refiner=args.train_refiner,
    )

    opt = th.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    baseline_val = eval_buckets(model, val_dl, device, "val")
    baseline_val.to_csv(out_dir / "r146_val_epoch0_before.csv", index=False)

    print("\n===== R146 BEFORE VAL =====")
    print("[R146] global cos:", global_metric(baseline_val, "cos"))
    print("[R146] global jss:", global_metric(baseline_val, "jss"))
    print(baseline_val.to_string(index=False))

    best_score = -1e18
    best_path = out_dir / "r146_best_state.pt"
    train_logs = []

    for epoch in range(1, args.epochs + 1):
        set_k3b_train_mode(
            model,
            train_formula_module=args.train_formula_module,
            train_refiner=args.train_refiner,
        )

        losses = []
        pbar = tqdm(train_dl, desc=f"R146 train epoch={epoch}")
        for bi, batch in enumerate(pbar):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break

            batch = move_to_device(batch, device)
            res = model._common_step(batch, split="train", log=False)
            loss = res["mean_loss"]

            if not th.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                5.0,
            )
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=np.mean(losses[-20:]))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "n_steps": len(losses),
        }
        train_logs.append(row)
        print("[R146 train]", row)

        val_sum = eval_buckets(model, val_dl, device, "val")
        val_sum.to_csv(out_dir / f"r146_val_epoch{epoch}.csv", index=False)

        score = score_val(val_sum, baseline_val)
        print(f"[R146] epoch={epoch} score={score:.6f}")
        print("[R146] global cos:", global_metric(val_sum, "cos"))
        print("[R146] global jss:", global_metric(val_sum, "jss"))
        print(val_sum.to_string(index=False))

        if score > best_score:
            best_score = score
            th.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "best_score": best_score,
            }, best_path)
            print("[R146] saved best:", best_path)

    pd.DataFrame(train_logs).to_csv(out_dir / "r146_train_log.csv", index=False)

    if best_path.exists():
        ckpt = th.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("[R146] loaded best epoch:", ckpt["epoch"], "score:", ckpt["best_score"])

    best_val = eval_buckets(model, val_dl, device, "val")
    best_val.to_csv(out_dir / "r146_best_val.csv", index=False)

    print("\n===== R146 BEST VAL =====")
    print("[R146] best global cos:", global_metric(best_val, "cos"))
    print("[R146] best global jss:", global_metric(best_val, "jss"))
    print(best_val.to_string(index=False))

    if args.eval_test:
        best_test = eval_buckets(model, test_dl, device, "test")
        best_test.to_csv(out_dir / "r146_best_test.csv", index=False)

        print("\n===== R146 BEST TEST =====")
        print("[R146] best global cos:", global_metric(best_test, "cos"))
        print("[R146] best global jss:", global_metric(best_test, "jss"))
        print(best_test.to_string(index=False))

    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
