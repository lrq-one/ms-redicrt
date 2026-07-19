import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from ms2spectra.workflow import load_config, init_dataset, init_dataloader
from ms2spectra.training import FragGNNPL


from train._impl.refinement_steps import (
    formula_composition,
)

def override_cfg_r147(cfg, args):
    # ===== K3b: keep the static formula/node residual path enabled =====
    cfg["use_formula_comp_residual"] = True
    cfg["formula_comp_hidden_size"] = int(args.k3b_hidden)
    cfg["formula_comp_dropout"] = float(args.k3b_dropout)
    cfg["formula_comp_delta_scale"] = float(args.k3b_delta_scale)
    cfg["formula_comp_center_per_spectrum"] = True
    cfg["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    # Dataset side: required by both K3b and CE-response when formula_comp is used.
    cfg.setdefault("frag_params", {})
    cfg["frag_params"]["formula_comp_feats"] = True
    cfg["frag_params"]["formula_comp_feat_size"] = int(args.formula_comp_feat_size)

    # ===== R147: CE-response joint residual =====
    # This is the important new part:
    # delta(node, formula/H, CE) is added to frag_joint_logits before R12 / softmax / rendering.
    cfg["use_ce_response_scorer"] = True
    cfg["ce_response_hidden_size"] = int(args.ce_hidden)
    cfg["ce_response_dropout"] = float(args.ce_dropout)
    cfg["ce_response_delta_scale"] = float(args.ce_delta_scale)
    cfg["ce_response_center_per_spectrum"] = True
    cfg["ce_response_use_formula_comp"] = bool(args.ce_use_formula_comp)
    cfg["ce_response_use_depth"] = bool(args.ce_use_depth)
    cfg["ce_response_use_h"] = bool(args.ce_use_h)

    # ===== R154: CEFlowFrag v2 =====
    # Multi-path CE-conditioned fragmentation flow prior.
    cfg["use_ce_flowfrag"] = bool(getattr(args, "use_ce_flowfrag", False))
    cfg["ce_flowfrag_lambda_max"] = float(getattr(args, "ce_flowfrag_lambda_max", 0.0))
    cfg["ce_flowfrag_hidden_size"] = int(getattr(args, "ce_flowfrag_hidden", 128))
    cfg["ce_flowfrag_dropout"] = float(getattr(args, "ce_flowfrag_dropout", 0.05))
    cfg["ce_flowfrag_max_depth"] = int(getattr(args, "ce_flowfrag_max_depth", 4))
    cfg["ce_flowfrag_mixture_hidden_size"] = int(getattr(args, "ce_flowfrag_mixture_hidden", 128))
    cfg["ce_flowfrag_mixture_dropout"] = float(getattr(args, "ce_flowfrag_mixture_dropout", 0.05))
    cfg["ce_flowfrag_mixture_init_bias"] = float(getattr(args, "ce_flowfrag_mixture_init_bias", -4.0))
    cfg["ce_flowfrag_delta_clip"] = float(getattr(args, "ce_flowfrag_delta_clip", 3.0))
    cfg["ce_flowfrag_use_direct_node"] = bool(getattr(args, "ce_flowfrag_use_direct_node", True))
    cfg["ce_flowfrag_direct_mix"] = float(getattr(args, "ce_flowfrag_direct_mix", 0.25))

    # ===== real official-path renderer during train/eval =====
    cfg["use_binned_spectrum_renderer"] = True
    cfg["binned_spectrum_renderer_apply_train"] = True
    cfg["binned_spectrum_renderer_bin_res"] = float(args.bin_res)
    cfg["binned_spectrum_renderer_preserve_mass"] = True
    cfg["binned_spectrum_renderer_max_bins"] = int(args.max_bins)

    # ===== CE-weighted binned aux loss =====
    cfg["use_ce_weighted_binned_aux_loss"] = bool(args.ce_binned_aux_weight > 0)
    cfg["ce_binned_aux_loss_weight"] = float(args.ce_binned_aux_weight)
    cfg["ce_binned_aux_mid_threshold"] = 20.0
    cfg["ce_binned_aux_high_threshold"] = 40.0
    cfg["ce_binned_aux_low_weight"] = float(args.low_w)
    cfg["ce_binned_aux_mid_weight"] = float(args.mid_w)
    cfg["ce_binned_aux_high_weight"] = float(args.high_w)

    # Optional fixed-support intensity allocation loss.
    # First round keep off. It can be enabled later if CE-response is stable.
    cfg["use_r117_support_oracle_reweight_loss"] = bool(args.r117_weight > 0)
    cfg["r117_support_oracle_weight"] = float(args.r117_weight)
    cfg["r117_oracle_bin_res"] = float(args.bin_res)
    cfg["r117_false_mass_weight"] = float(args.r117_false_weight)
    cfg["r117_min_covered_true_mass"] = 1.0e-8
    cfg["r117_eps"] = 1.0e-12

    return cfg


def freeze_for_r147(model, train_k3b=False, train_refiner=False, train_formula_module=False, train_render_gate=False, train_frag_rep=False, train_ce_flowfrag=False):
    for p in model.parameters():
        p.requires_grad_(False)

    allow = ["ce_response_scorer"]

    if train_k3b:
        allow.append("formula_comp_residual_head")
    if train_refiner:
        allow.append("spectrum_candidate_refiner")
    if train_formula_module:
        allow.append("formula_module")
    if train_render_gate:
        allow.append("rendered_peak_drop_gate")
    if train_frag_rep:
        allow.extend([
            "frag_embedder",
            "frag_pool",
            "formula_embedder",
            "depth_embedder",
            "complement_embedder",
            "cc_interstage",
        ])
    if train_ce_flowfrag:
        allow.extend([
            "ce_flowfrag_edge_head",
            "ce_flowfrag_direct_head",
            "ce_flowfrag_mixture_head",
        ])

    train_names = []
    for name, p in model.model.named_parameters():
        if any(name.startswith(prefix) for prefix in allow):
            p.requires_grad_(True)
            train_names.append(name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    print("[R147] trainable params:", n_train, "/", n_total)
    print("[R147] trainable prefixes:", allow)
    print("[R147] first trainable names:")
    for n in train_names[:40]:
        print("  ", n)
    if len(train_names) > 40:
        print("  ...")

    if n_train == 0:
        raise RuntimeError("No trainable params. ce_response_scorer not found.")


def set_r147_train_mode(model, train_k3b=False, train_refiner=False, train_formula_module=False, train_render_gate=False, train_frag_rep=False, train_ce_flowfrag=False):
    # Frozen backbone eval mode, only selected residual heads in train mode.
    model.eval()

    for name, module in model.model.named_modules():
        if name.startswith("ce_response_scorer"):
            module.train()
        if train_k3b and name.startswith("formula_comp_residual_head"):
            module.train()
        if train_refiner and name.startswith("spectrum_candidate_refiner"):
            module.train()
        if train_formula_module and name.startswith("formula_module"):
            module.train()
        if train_render_gate and name.startswith("rendered_peak_drop_gate"):
            module.train()
        if train_frag_rep and (
            name.startswith("frag_embedder")
            or name.startswith("frag_pool")
            or name.startswith("formula_embedder")
            or name.startswith("depth_embedder")
            or name.startswith("complement_embedder")
            or name.startswith("cc_interstage")
        ):
            module.train()
        if train_ce_flowfrag and (
            name.startswith("ce_flowfrag_edge_head")
            or name.startswith("ce_flowfrag_direct_head")
            or name.startswith("ce_flowfrag_mixture_head")
        ):
            module.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_train_batches", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    # K3b config must match the checkpoint if loading from R146 best.
    ap.add_argument("--k3b_hidden", type=int, default=128)
    ap.add_argument("--k3b_dropout", type=float, default=0.05)
    ap.add_argument("--k3b_delta_scale", type=float, default=0.05)
    ap.add_argument("--formula_comp_feat_size", type=int, default=18)

    # CE-response config.
    ap.add_argument("--ce_hidden", type=int, default=128)
    ap.add_argument("--ce_dropout", type=float, default=0.05)
    ap.add_argument("--ce_delta_scale", type=float, default=0.025)
    ap.add_argument("--ce_use_formula_comp", action="store_true")
    ap.add_argument("--ce_use_depth", action="store_true")
    ap.add_argument("--ce_use_h", action="store_true")

    # CEFlowFrag v2.
    ap.add_argument("--use_ce_flowfrag", action="store_true")
    ap.add_argument("--ce_flowfrag_lambda_max", type=float, default=0.0)
    ap.add_argument("--ce_flowfrag_hidden", type=int, default=128)
    ap.add_argument("--ce_flowfrag_dropout", type=float, default=0.05)
    ap.add_argument("--ce_flowfrag_max_depth", type=int, default=4)
    ap.add_argument("--ce_flowfrag_mixture_hidden", type=int, default=128)
    ap.add_argument("--ce_flowfrag_mixture_dropout", type=float, default=0.05)
    ap.add_argument("--ce_flowfrag_mixture_init_bias", type=float, default=-4.0)
    ap.add_argument("--ce_flowfrag_delta_clip", type=float, default=3.0)
    ap.add_argument("--ce_flowfrag_use_direct_node", action="store_true")
    ap.add_argument("--ce_flowfrag_direct_mix", type=float, default=0.25)

    ap.add_argument("--bin_res", type=float, default=0.01)
    ap.add_argument("--max_bins", type=int, default=0)

    ap.add_argument("--ce_binned_aux_weight", type=float, default=0.0015)
    ap.add_argument("--r117_weight", type=float, default=0.0)
    ap.add_argument("--r117_false_weight", type=float, default=0.20)

    # Slightly stronger mid/high emphasis than R146.
    ap.add_argument("--low_w", type=float, default=0.25)
    ap.add_argument("--mid_w", type=float, default=1.75)
    ap.add_argument("--high_w", type=float, default=2.25)

    # Train switches.
    ap.add_argument("--train_k3b", action="store_true")
    ap.add_argument("--train_refiner", action="store_true")
    ap.add_argument("--train_formula_module", action="store_true")
    ap.add_argument("--train_render_gate", action="store_true")
    ap.add_argument("--train_frag_rep", action="store_true")
    ap.add_argument("--train_ce_flowfrag", action="store_true")

    ap.add_argument("--eval_test", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(args)]).to_csv(out_dir / "r147_args.csv", index=False)

    cfg = load_config(args.template, args.config)
    cfg = override_cfg_r147(cfg, args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    model = FragGNNPL(**cfg)
    sd = formula_composition.load_state_dict(args.ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("[R147] missing keys:", len(missing))
    for k in missing[:40]:
        print("  missing:", k)
    if len(missing) > 40:
        print("  ...")
    print("[R147] unexpected keys:", len(unexpected))
    for k in unexpected[:20]:
        print("  unexpected:", k)

    model = model.to(device)

    freeze_for_r147(
        model,
        train_k3b=args.train_k3b,
        train_refiner=args.train_refiner,
        train_formula_module=args.train_formula_module,
        train_render_gate=args.train_render_gate,
        train_frag_rep=args.train_frag_rep,
        train_ce_flowfrag=args.train_ce_flowfrag,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = th.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    baseline_val = formula_composition.eval_buckets(model, val_dl, device, "val")
    baseline_val.to_csv(out_dir / "r147_val_epoch0_before.csv", index=False)

    print("\n===== R147 BEFORE VAL =====")
    print("[R147] global cos:", formula_composition.global_metric(baseline_val, "cos"))
    print("[R147] global jss:", formula_composition.global_metric(baseline_val, "jss"))
    print(baseline_val.to_string(index=False))

    best_score = -1e18
    best_path = out_dir / "r147_best_state.pt"
    logs = []

    for epoch in range(1, args.epochs + 1):
        set_r147_train_mode(
            model,
            train_k3b=args.train_k3b,
            train_refiner=args.train_refiner,
            train_formula_module=args.train_formula_module,
        )

        losses = []
        pbar = tqdm(train_dl, desc=f"R147 train epoch={epoch}")

        for bi, batch in enumerate(pbar):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break

            batch = formula_composition.move_to_device(batch, device)
            res = model._common_step(batch, split="train", log=False)
            loss = res["mean_loss"]

            if not th.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            pbar.set_postfix(loss=np.mean(losses[-20:]))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "n_steps": len(losses),
        }
        logs.append(row)
        print("[R147 train]", row)

        val_sum = formula_composition.eval_buckets(model, val_dl, device, "val")
        val_sum.to_csv(out_dir / f"r147_val_epoch{epoch}.csv", index=False)

        score = formula_composition.score_val(val_sum, baseline_val)
        print(f"[R147] epoch={epoch} score={score:.6f}")
        print("[R147] global cos:", formula_composition.global_metric(val_sum, "cos"))
        print("[R147] global jss:", formula_composition.global_metric(val_sum, "jss"))
        print(val_sum.to_string(index=False))

        if score > best_score:
            best_score = score
            th.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_score": best_score,
                },
                best_path,
            )
            print("[R147] saved best:", best_path)

    pd.DataFrame(logs).to_csv(out_dir / "r147_train_log.csv", index=False)

    if best_path.exists():
        ckpt = th.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("[R147] loaded best epoch:", ckpt["epoch"], "score:", ckpt["best_score"])

    best_val = formula_composition.eval_buckets(model, val_dl, device, "val")
    best_val.to_csv(out_dir / "r147_best_val.csv", index=False)

    print("\n===== R147 BEST VAL =====")
    print("[R147] best global cos:", formula_composition.global_metric(best_val, "cos"))
    print("[R147] best global jss:", formula_composition.global_metric(best_val, "jss"))
    print(best_val.to_string(index=False))

    if args.eval_test:
        best_test = formula_composition.eval_buckets(model, test_dl, device, "test")
        best_test.to_csv(out_dir / "r147_best_test.csv", index=False)

        print("\n===== R147 BEST TEST =====")
        print("[R147] best global cos:", formula_composition.global_metric(best_test, "cos"))
        print("[R147] best global jss:", formula_composition.global_metric(best_test, "jss"))
        print(best_test.to_string(index=False))

    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
