import argparse
import importlib.util
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ms2spectra.workflow import load_config, init_dataset, init_dataloader
from ms2spectra.training import FragGNNPL


def load_r170_module():
    fp = Path(__file__).with_name("candidate_reranker.py")
    spec = importlib.util.spec_from_file_location("r170_mod", str(fp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ResidualAllocator(nn.Module):
    """
    Zero-init residual allocator.

    At init:
        residual = 0
        new_logp = softmax(old_logp + alpha * LGBM_score)

    So BEFORE VAL/TEST should match R172E.
    """
    def __init__(self, input_dim, hidden=256, layers=3, dropout=0.10, score_clip=4.0):
        super().__init__()
        self.score_clip = float(score_clip)

        mods = []
        dim = int(input_dim)
        for _ in range(int(layers)):
            mods.append(nn.Linear(dim, int(hidden)))
            mods.append(nn.LayerNorm(int(hidden)))
            mods.append(nn.GELU())
            mods.append(nn.Dropout(float(dropout)))
            dim = int(hidden)

        final = nn.Linear(dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        mods.append(final)

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        y = self.net(x).squeeze(-1)
        return y.clamp(-self.score_clip, self.score_clip)


def group_log_softmax(scores, bidx, batch_size):
    out = th.empty_like(scores)
    for b in range(int(batch_size)):
        m = bidx == b
        if m.any():
            out[m] = F.log_softmax(scores[m], dim=0)
    return out


def group_sum(vals, bidx, batch_size):
    out = th.zeros(int(batch_size), device=vals.device, dtype=vals.dtype)
    out.index_add_(0, bidx.long(), vals)
    return out


def ce_weights(ce, low_w, mid_w, high_w):
    w = th.ones_like(ce, dtype=th.float32)
    w = th.where(ce <= 20.0, th.full_like(w, float(low_w)), w)
    w = th.where((ce > 20.0) & (ce <= 40.0), th.full_like(w, float(mid_w)), w)
    w = th.where(ce > 40.0, th.full_like(w, float(high_w)), w)
    return w / w.mean().clamp_min(1e-12)


def dense_by_round_bins_grad(mzs, vals, batch_idxs, batch_size, mz_max, bin_res):
    device = mzs.device
    dtype = vals.dtype
    num_bins = int(float(mz_max) / float(bin_res)) + 2
    dense = th.zeros((int(batch_size), num_bins), device=device, dtype=dtype)

    if mzs.numel() == 0:
        return dense

    bins = th.round(mzs.float() / float(bin_res)).long().clamp(0, num_bins - 1)
    bidx = batch_idxs.long().clamp(0, int(batch_size) - 1)
    flat = bidx * num_bins + bins

    dense_flat = dense.reshape(-1)
    dense_flat.index_add_(0, flat, vals)
    return dense_flat.reshape(int(batch_size), num_bins)


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


def normalize_target_mass_per_spec(target_mass, bidx, batch_size):
    target_mass = target_mass.clamp_min(0.0)
    denom = group_sum(target_mass, bidx, batch_size).clamp_min(1e-12)
    return target_mass / denom[bidx.long()]


def alias_rich_feature_keys(res, extra_schema):
    """
    当前 diagnostics/87 可能输出 r173_frag_rich_feats；
    老的 R172D regressor pack 可能记录的是 r172_frag_rich_feats。
    这里做别名，避免 regressor 输入 60D 里 26D rich features 变成全 0。
    """
    keys = {k for k, _ in extra_schema}

    if "r172_frag_rich_feats" in keys:
        if "r172_frag_rich_feats" not in res and "r173_frag_rich_feats" in res:
            res["r172_frag_rich_feats"] = res["r173_frag_rich_feats"]

    if "r173_frag_rich_feats" in keys:
        if "r173_frag_rich_feats" not in res and "r172_frag_rich_feats" in res:
            res["r173_frag_rich_feats"] = res["r172_frag_rich_feats"]

    return res


def lgbm_predict(regressor, feats, device, dtype, score_clip):
    X = feats.detach().cpu().numpy().astype(np.float32)
    s = regressor.predict(X).astype(np.float32)
    s = np.clip(s, -float(score_clip), float(score_clip))
    return th.from_numpy(s).to(device=device, dtype=dtype)


def build_batch_tensors(base, batch, r170, regressor, extra_schema, args, split):
    with th.no_grad():
        res = base._common_step(batch, split=split, log=False)

        res = r170.attach_raw_rich_features(
            base,
            batch,
            res,
            max_extra_dims=int(args.max_extra_dims),
            bin_res=float(args.target_bin_res),
        )
        res = alias_rich_feature_keys(res, extra_schema)

        feats = r170.candidate_features(
            res,
            batch,
            mz_max=float(base.hparams.mz_max),
            local_bin_res=float(args.local_bin_res),
            extra_schema=extra_schema,
        )

        args.mz_max = float(base.hparams.mz_max)
        residual_target, pos, target_mass, pred_prob = r170.build_targets(res, args)

        lgbm_score = lgbm_predict(
            regressor,
            feats,
            device=feats.device,
            dtype=res["pred_logprobs"].dtype,
            score_clip=float(args.lgbm_score_clip),
        )

    return res, feats.float(), lgbm_score.float(), target_mass.float(), pos.float()


def forward_allocator(base, allocator, batch, res, feats, lgbm_score, target_mass, r170, args):
    pred_mz = res["pred_mzs"].float()
    old_logp = res["pred_logprobs"].float()
    bidx = res["pred_batch_idxs"].long()
    batch_size = int(res["unique_id"].numel())

    true_dense = dense_by_round_bins_grad(
        res["true_mzs"].float(),
        res["true_logprobs"].exp().float(),
        res["true_batch_idxs"].long(),
        batch_size=batch_size,
        mz_max=float(base.hparams.mz_max),
        bin_res=float(args.eval_bin_res),
    )

    base_logits = old_logp.clamp(-30.0, 5.0) + float(args.alpha) * lgbm_score
    base_logp = group_log_softmax(base_logits, bidx, batch_size)

    residual = allocator(feats)

    logits = base_logits + float(args.residual_scale) * residual
    logits = logits / max(float(args.temperature), 1e-6)

    new_logp = group_log_softmax(logits, bidx, batch_size)
    new_prob = new_logp.exp()

    pred_dense = dense_by_round_bins_grad(
        pred_mz,
        new_prob,
        bidx,
        batch_size=batch_size,
        mz_max=float(base.hparams.mz_max),
        bin_res=float(args.eval_bin_res),
    )

    cos = cosine_dense(true_dense, pred_dense).clamp(0.0, 1.0)
    jss = jss_dense(true_dense, pred_dense).clamp(0.0, 1.0)

    ce, _ = r170.find_ce(batch)
    ce = ce.to(pred_mz.device).reshape(-1).float()
    spec_w = ce_weights(ce, args.low_w, args.mid_w, args.high_w)

    spec_loss_vec = (
        float(args.cos_weight) * (1.0 - cos)
        + float(args.jss_weight) * (1.0 - jss)
    )
    spec_loss = (spec_w * spec_loss_vec).sum() / spec_w.sum().clamp_min(1e-12)

    # target CE is now small. R183A failed partly because target CE dominated.
    target_dist = normalize_target_mass_per_spec(target_mass, bidx, batch_size)
    target_ce_per_entry = -(target_dist.detach() * new_logp)
    target_ce_per_spec = group_sum(target_ce_per_entry, bidx, batch_size)
    target_ce = (spec_w * target_ce_per_spec).sum() / spec_w.sum().clamp_min(1e-12)

    pos_mask = target_mass > 0
    pos_mass = th.zeros(batch_size, device=pred_mz.device, dtype=new_prob.dtype)
    if pos_mask.any():
        pos_mass.index_add_(0, bidx[pos_mask], new_prob[pos_mask])
    pos_recall_loss = (spec_w * (1.0 - pos_mass.clamp(0.0, 1.0))).sum() / spec_w.sum().clamp_min(1e-12)

    base_prob = base_logp.exp().detach()
    kl_entry = new_prob * (new_logp - base_logp.detach())
    kl_per_spec = group_sum(kl_entry, bidx, batch_size)
    base_kl = (spec_w * kl_per_spec).sum() / spec_w.sum().clamp_min(1e-12)

    residual_l2 = residual.pow(2).mean()

    loss = (
        spec_loss
        + float(args.target_ce_weight) * target_ce
        + float(args.pos_recall_weight) * pos_recall_loss
        + float(args.base_kl_weight) * base_kl
        + float(args.residual_l2_weight) * residual_l2
    )

    return {
        "loss": loss,
        "new_logp": new_logp,
        "base_logp": base_logp,
        "cos": cos.detach(),
        "jss": jss.detach(),
        "spec_loss": spec_loss.detach(),
        "target_ce": target_ce.detach(),
        "pos_recall_loss": pos_recall_loss.detach(),
        "base_kl": base_kl.detach(),
        "residual_l2": residual_l2.detach(),
        "residual_mean_abs": residual.abs().mean().detach(),
    }


def summarize_rows(rows):
    df = pd.DataFrame(rows)
    out = []

    for bucket in ["global", "low_<=20", "mid_20_40", "high_>40"]:
        sub = df if bucket == "global" else df[df["ce_bucket"] == bucket]
        if len(sub) == 0:
            continue

        out.append({
            "ce_bucket": bucket,
            "spec_count": int(len(sub)),
            "mean_ce": float(sub["ce"].mean()),
            "cos": float(sub["cos"].mean()),
            "jss": float(sub["jss"].mean()),
        })

    return pd.DataFrame(out), df


@th.no_grad()
def eval_split(base, allocator, regressor, extra_schema, dl, device, r170, args, split):
    base.eval()
    allocator.eval()

    rows = []

    for batch in tqdm(dl, desc=f"eval {split}"):
        batch = r170.move_to_device(batch, device)

        res, feats, lgbm_score, target_mass, pos = build_batch_tensors(
            base,
            batch,
            r170,
            regressor,
            extra_schema,
            args,
            split=split,
        )

        out = forward_allocator(
            base,
            allocator,
            batch,
            res,
            feats,
            lgbm_score,
            target_mass,
            r170,
            args,
        )

        ce, _ = r170.find_ce(batch)
        ce_cpu = ce.detach().cpu().reshape(-1)
        buckets = r170.ce_bucket_names(ce_cpu)

        for i, sid in enumerate(res["unique_id"].detach().cpu().reshape(-1).numpy().astype(int)):
            rows.append({
                "spec_id": int(sid),
                "ce": float(ce_cpu[i]),
                "ce_bucket": buckets[i],
                "cos": float(out["cos"][i].detach().cpu()),
                "jss": float(out["jss"][i].detach().cpu()),
            })

    return summarize_rows(rows)


def train_one_epoch(base, allocator, regressor, extra_schema, train_dl, opt, device, r170, args, epoch):
    base.eval()
    allocator.train()

    sums = {
        "loss": 0.0,
        "cos": 0.0,
        "jss": 0.0,
        "spec_loss": 0.0,
        "target_ce": 0.0,
        "pos_recall_loss": 0.0,
        "base_kl": 0.0,
        "residual_l2": 0.0,
        "residual_mean_abs": 0.0,
        "n": 0,
    }

    pbar = tqdm(train_dl, desc=f"R184 train epoch={epoch}")

    for step, batch in enumerate(pbar, start=1):
        if int(args.max_train_batches) > 0 and step > int(args.max_train_batches):
            break

        batch = r170.move_to_device(batch, device)

        res, feats, lgbm_score, target_mass, pos = build_batch_tensors(
            base,
            batch,
            r170,
            regressor,
            extra_schema,
            args,
            split="train",
        )

        out = forward_allocator(
            base,
            allocator,
            batch,
            res,
            feats,
            lgbm_score,
            target_mass,
            r170,
            args,
        )

        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(allocator.parameters(), float(args.grad_clip))
        opt.step()

        bs = int(res["unique_id"].numel())

        for k in [
            "loss",
            "spec_loss",
            "target_ce",
            "pos_recall_loss",
            "base_kl",
            "residual_l2",
            "residual_mean_abs",
        ]:
            sums[k] += float(out[k].detach().cpu()) * bs

        sums["cos"] += float(out["cos"].mean().detach().cpu()) * bs
        sums["jss"] += float(out["jss"].mean().detach().cpu()) * bs
        sums["n"] += bs

        pbar.set_postfix({
            "loss": f"{float(out['loss'].detach().cpu()):.4f}",
            "cos": f"{float(out['cos'].mean().detach().cpu()):.4f}",
            "jss": f"{float(out['jss'].mean().detach().cpu()):.4f}",
            "res": f"{float(out['residual_mean_abs'].detach().cpu()):.3f}",
        })

    n = max(1, int(sums["n"]))
    return {k: (v / n if k != "n" else v) for k, v in sums.items()}


def save_pack(path, allocator, input_dim, extra_schema, args, best_val_cos):
    th.save({
        "model": allocator.state_dict(),
        "input_dim": int(input_dim),
        "extra_schema": extra_schema,
        "args": vars(args),
        "best_val_cos": float(best_val_cos),
    }, path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--regressor_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--score_clip", type=float, default=4.0)

    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--lgbm_score_clip", type=float, default=6.0)
    ap.add_argument("--residual_scale", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--cos_weight", type=float, default=1.0)
    ap.add_argument("--jss_weight", type=float, default=0.30)
    ap.add_argument("--target_ce_weight", type=float, default=0.005)
    ap.add_argument("--pos_recall_weight", type=float, default=0.02)
    ap.add_argument("--base_kl_weight", type=float, default=0.04)
    ap.add_argument("--residual_l2_weight", type=float, default=0.001)

    ap.add_argument("--low_w", type=float, default=0.50)
    ap.add_argument("--mid_w", type=float, default=2.50)
    ap.add_argument("--high_w", type=float, default=5.00)

    ap.add_argument("--mz_tol", type=float, default=0.01)
    ap.add_argument("--mz_sigma", type=float, default=0.003)
    ap.add_argument("--target_bin_res", type=float, default=0.01)
    ap.add_argument("--local_bin_res", type=float, default=0.01)
    ap.add_argument("--eval_bin_res", type=float, default=0.01)
    ap.add_argument("--residual_clip", type=float, default=6.0)
    ap.add_argument("--neg_residual", type=float, default=4.0)

    ap.add_argument("--max_extra_dims", type=int, default=26)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--eval_test", action="store_true")

    args = ap.parse_args()

    np.random.seed(int(args.seed))
    th.manual_seed(int(args.seed))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r170 = load_r170_module()

    with open(args.regressor_path, "rb") as f:
        reg_pack = pickle.load(f)

    regressor = reg_pack["model"]
    extra_schema = reg_pack.get("extra_schema", [])
    backend = reg_pack.get("backend", "unknown")

    print("[R184] loaded regressor:", args.regressor_path)
    print("[R184] regressor backend:", backend)
    print("[R184] regressor extra_schema:", extra_schema)

    cfg = load_config(args.template, args.config)
    cfg = r170.force_r160_arch(cfg)

    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    base = FragGNNPL(**cfg)
    sd = r170.load_state_dict_any(args.ckpt_path)
    missing, unexpected = base.load_state_dict(sd, strict=False)

    print("[R184] base missing:", len(missing))
    for x in missing[:20]:
        print("  missing:", x)
    print("[R184] base unexpected:", len(unexpected))
    for x in unexpected[:20]:
        print("  unexpected:", x)

    base = base.to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    first_batch = next(iter(train_dl))
    first_batch = r170.move_to_device(first_batch, device)

    with th.no_grad():
        first_res = base._common_step(first_batch, split="train", log=False)
        first_res = r170.attach_raw_rich_features(
            base,
            first_batch,
            first_res,
            max_extra_dims=int(args.max_extra_dims),
            bin_res=float(args.target_bin_res),
        )
        first_res = alias_rich_feature_keys(first_res, extra_schema)

        first_feats = r170.candidate_features(
            first_res,
            first_batch,
            mz_max=float(base.hparams.mz_max),
            local_bin_res=float(args.local_bin_res),
            extra_schema=extra_schema,
        )

    input_dim = int(first_feats.shape[1])
    print("[R184] input_dim:", input_dim)

    if hasattr(regressor, "n_features_in_"):
        print("[R184] regressor n_features_in_:", int(regressor.n_features_in_))
        if int(regressor.n_features_in_) != input_dim:
            raise RuntimeError(
                f"feature dim mismatch: allocator/input_dim={input_dim}, "
                f"regressor.n_features_in_={int(regressor.n_features_in_)}"
            )

    allocator = ResidualAllocator(
        input_dim=input_dim,
        hidden=int(args.hidden),
        layers=int(args.layers),
        dropout=float(args.dropout),
        score_clip=float(args.score_clip),
    ).to(device)

    opt = th.optim.AdamW(
        allocator.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_rows = []

    # Initial evaluation: should match R172E alpha result.
    before_val, before_val_detail = eval_split(
        base,
        allocator,
        regressor,
        extra_schema,
        val_dl,
        device,
        r170,
        args,
        split="val",
    )
    before_val.to_csv(out_dir / "r184_val_epoch0_before.csv", index=False)

    print("\n===== R184 BEFORE VAL =====")
    print(before_val.to_string(index=False))

    best_val_cos = float(before_val[before_val["ce_bucket"] == "global"].iloc[0]["cos"])
    save_pack(
        out_dir / "r184_allocator_best.pt",
        allocator,
        input_dim,
        extra_schema,
        args,
        best_val_cos,
    )
    before_val.to_csv(out_dir / "r184_best_val.csv", index=False)
    print("[R184] saved initial best:", best_val_cos)

    for epoch in range(1, int(args.epochs) + 1):
        tr = train_one_epoch(
            base,
            allocator,
            regressor,
            extra_schema,
            train_dl,
            opt,
            device,
            r170,
            args,
            epoch,
        )
        tr["epoch"] = epoch
        train_rows.append(tr)
        pd.DataFrame(train_rows).to_csv(out_dir / "r184_train_log.csv", index=False)

        val_table, val_detail = eval_split(
            base,
            allocator,
            regressor,
            extra_schema,
            val_dl,
            device,
            r170,
            args,
            split="val",
        )
        val_table.to_csv(out_dir / f"r184_val_epoch{epoch}.csv", index=False)

        g = val_table[val_table["ce_bucket"] == "global"].iloc[0]
        val_cos = float(g["cos"])

        print(f"\n===== R184 VAL epoch={epoch} score={val_cos:.6f} =====")
        print(val_table.to_string(index=False))

        if val_cos > best_val_cos:
            best_val_cos = val_cos
            save_pack(
                out_dir / "r184_allocator_best.pt",
                allocator,
                input_dim,
                extra_schema,
                args,
                best_val_cos,
            )
            val_table.to_csv(out_dir / "r184_best_val.csv", index=False)
            print("[R184] saved best:", out_dir / "r184_allocator_best.pt")

    pack = th.load(out_dir / "r184_allocator_best.pt", map_location=device, weights_only=False)
    allocator.load_state_dict(pack["model"])
    allocator.eval()

    best_val, best_val_detail = eval_split(
        base,
        allocator,
        regressor,
        extra_schema,
        val_dl,
        device,
        r170,
        args,
        split="val",
    )
    best_val.to_csv(out_dir / "r184_best_val.csv", index=False)

    print("\n===== R184 BEST VAL =====")
    print(best_val.to_string(index=False))

    if args.eval_test:
        best_test, best_test_detail = eval_split(
            base,
            allocator,
            regressor,
            extra_schema,
            test_dl,
            device,
            r170,
            args,
            split="test",
        )
        best_test.to_csv(out_dir / "r184_best_test.csv", index=False)
        best_test_detail.to_csv(out_dir / "r184_best_test_detail.csv", index=False)

        print("\n===== R184 BEST TEST =====")
        print(best_test.to_string(index=False))

    print("wrote", out_dir)


if __name__ == "__main__":
    main()
