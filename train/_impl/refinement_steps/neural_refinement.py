import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from ms2spectra.workflow import load_config, init_dataset, init_dataloader
from ms2spectra.training import FragGNNPL


from train._impl.refinement_steps import (
    collision_energy_response
    as energy_response,
)

from train._impl.refinement_steps import formula_composition

from train._impl.refinement_steps.formula_supervision import (
    build_target_formula_hard,
    formula_losses as compute_formula_losses,
    get_batch_size,
    get_spec_ce,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--template", default="runs/_config/template.yml")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_train_batches", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)

    # Must match loaded R147/R146 architecture.
    ap.add_argument("--k3b_hidden", type=int, default=128)
    ap.add_argument("--k3b_dropout", type=float, default=0.05)
    ap.add_argument("--k3b_delta_scale", type=float, default=0.05)
    ap.add_argument("--formula_comp_feat_size", type=int, default=18)

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

    # Official-path spectrum loss settings.
    ap.add_argument("--ce_binned_aux_weight", type=float, default=0.0015)
    ap.add_argument("--r117_weight", type=float, default=0.0)
    ap.add_argument("--r117_false_weight", type=float, default=0.20)

    ap.add_argument("--low_w", type=float, default=0.25)
    ap.add_argument("--mid_w", type=float, default=1.75)
    ap.add_argument("--high_w", type=float, default=2.25)

    # Formula target aux.
    ap.add_argument("--formula_aux_weight", type=float, default=0.003)
    ap.add_argument("--formula_tol", type=float, default=0.01)
    ap.add_argument("--formula_mz_sigma", type=float, default=0.003)
    ap.add_argument("--hard_formula_topk", type=int, default=3)
    ap.add_argument("--formula_score_mode", default="max", choices=["max", "sum"])
    ap.add_argument("--prob_alpha", type=float, default=0.0)

    ap.add_argument("--formula_kl_weight", type=float, default=1.0)
    ap.add_argument("--formula_rank_weight", type=float, default=0.35)
    ap.add_argument("--formula_false_weight", type=float, default=0.03)
    ap.add_argument("--formula_target_topk", type=int, default=5)
    ap.add_argument("--formula_neg_topk", type=int, default=20)
    ap.add_argument("--formula_margin", type=float, default=0.50)
    ap.add_argument("--formula_neg_target_max", type=float, default=0.002)

    # CE weights for formula aux.
    ap.add_argument("--formula_low_w", type=float, default=0.05)
    ap.add_argument("--formula_mid_w", type=float, default=2.00)
    ap.add_argument("--formula_high_w", type=float, default=2.50)

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
    pd.DataFrame([vars(args)]).to_csv(out_dir / "r148_args.csv", index=False)

    cfg = load_config(args.template, args.config)
    cfg = energy_response.override_cfg_r147(cfg, args)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = init_dataset(cfg, splits=("train", "val", "test"))
    train_dl = init_dataloader(train_ds, cfg)
    val_dl = init_dataloader(val_ds, cfg)
    test_dl = init_dataloader(test_ds, cfg)

    model = FragGNNPL(**cfg)
    sd = formula_composition.load_state_dict(args.ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("[R148] missing keys:", len(missing))
    for k in missing[:40]:
        print("  missing:", k)
    if len(missing) > 40:
        print("  ...")
    print("[R148] unexpected keys:", len(unexpected))
    for k in unexpected[:20]:
        print("  unexpected:", k)

    model = model.to(device)

    energy_response.freeze_for_r147(
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
    baseline_val.to_csv(out_dir / "r148_val_epoch0_before.csv", index=False)

    print("\n===== R148 BEFORE VAL =====")
    print("[R148] global cos:", formula_composition.global_metric(baseline_val, "cos"))
    print("[R148] global jss:", formula_composition.global_metric(baseline_val, "jss"))
    print(baseline_val.to_string(index=False))

    best_score = -1e18
    best_path = out_dir / "r148_best_state.pt"
    logs = []

    for epoch in range(1, args.epochs + 1):
        energy_response.set_r147_train_mode(
            model,
            train_k3b=args.train_k3b,
            train_refiner=args.train_refiner,
            train_formula_module=args.train_formula_module,
        )

        losses = []
        base_losses = []
        formula_losses = []
        kl_losses = []
        rank_losses = []
        false_losses = []
        valid_fracs = []
        n_events = []

        pbar = tqdm(train_dl, desc=f"R148 train epoch={epoch}")

        for bi, batch in enumerate(pbar):
            if args.max_train_batches > 0 and bi >= args.max_train_batches:
                break

            batch = formula_composition.move_to_device(batch, device)
            bs = get_batch_size(batch)
            ce_np = get_spec_ce(batch, bs)

            # Real official-path spectrum loss.
            base_res = model._common_step(batch, split="train", log=False)
            base_loss = base_res["mean_loss"]

            # Separate forward for hard formula target aux.
            raw = model.forward(**batch)

            target_pack = build_target_formula_hard(
                model=model,
                batch=batch,
                raw=raw,
                tol=args.formula_tol,
                bin_res=args.bin_res,
                mz_sigma=args.formula_mz_sigma,
                hard_formula_topk=args.hard_formula_topk,
                formula_score_mode=args.formula_score_mode,
                prob_alpha=args.prob_alpha,
            )

            if target_pack is None:
                loss = base_loss
                f_loss = th.zeros((), dtype=base_loss.dtype, device=base_loss.device)
                stat = {
                    "kl_loss": np.nan,
                    "rank_loss": np.nan,
                    "false_loss": np.nan,
                    "valid_frac": 0.0,
                    "n_rank_specs": 0.0,
                }
                target_events = 0.0
            else:
                f_loss, stat = compute_formula_losses(
                    raw=raw,
                    target_formula=target_pack["target_formula"],
                    ce_np=ce_np,
                    low_w=args.formula_low_w,
                    mid_w=args.formula_mid_w,
                    high_w=args.formula_high_w,
                    kl_weight=args.formula_kl_weight,
                    rank_weight=args.formula_rank_weight,
                    false_weight=args.formula_false_weight,
                    target_topk=args.formula_target_topk,
                    neg_topk=args.formula_neg_topk,
                    margin=args.formula_margin,
                    neg_target_max=args.formula_neg_target_max,
                )
                target_events = float(target_pack.get("n_events", 0.0))
                loss = base_loss + float(args.formula_aux_weight) * f_loss

            if not th.isfinite(loss):
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))
            base_losses.append(float(base_loss.detach().cpu().item()))
            formula_losses.append(float(f_loss.detach().cpu().item()))
            kl_losses.append(float(stat["kl_loss"]) if np.isfinite(stat["kl_loss"]) else np.nan)
            rank_losses.append(float(stat["rank_loss"]) if np.isfinite(stat["rank_loss"]) else np.nan)
            false_losses.append(float(stat["false_loss"]) if np.isfinite(stat["false_loss"]) else np.nan)
            valid_fracs.append(float(stat["valid_frac"]))
            n_events.append(target_events)

            pbar.set_postfix(
                loss=np.mean(losses[-20:]),
                base=np.mean(base_losses[-20:]),
                f=np.mean(formula_losses[-20:]),
                events=np.mean(n_events[-20:]),
            )

        row = {
            "epoch": epoch,
            "train_loss": float(np.nanmean(losses)) if losses else np.nan,
            "base_loss": float(np.nanmean(base_losses)) if base_losses else np.nan,
            "formula_loss": float(np.nanmean(formula_losses)) if formula_losses else np.nan,
            "formula_kl_loss": float(np.nanmean(kl_losses)) if kl_losses else np.nan,
            "formula_rank_loss": float(np.nanmean(rank_losses)) if rank_losses else np.nan,
            "formula_false_loss": float(np.nanmean(false_losses)) if false_losses else np.nan,
            "formula_valid_frac": float(np.nanmean(valid_fracs)) if valid_fracs else np.nan,
            "n_events": float(np.nanmean(n_events)) if n_events else np.nan,
            "n_steps": len(losses),
        }
        logs.append(row)
        print("[R148 train]", row)

        val_sum = formula_composition.eval_buckets(model, val_dl, device, "val")
        val_sum.to_csv(out_dir / f"r148_val_epoch{epoch}.csv", index=False)

        score = formula_composition.score_val(val_sum, baseline_val)
        print(f"[R148] epoch={epoch} score={score:.6f}")
        print("[R148] global cos:", formula_composition.global_metric(val_sum, "cos"))
        print("[R148] global jss:", formula_composition.global_metric(val_sum, "jss"))
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
            print("[R148] saved best:", best_path)

    pd.DataFrame(logs).to_csv(out_dir / "r148_train_log.csv", index=False)

    if best_path.exists():
        ckpt = th.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("[R148] loaded best epoch:", ckpt["epoch"], "score:", ckpt["best_score"])

    best_val = formula_composition.eval_buckets(model, val_dl, device, "val")
    best_val.to_csv(out_dir / "r148_best_val.csv", index=False)

    print("\n===== R148 BEST VAL =====")
    print("[R148] best global cos:", formula_composition.global_metric(best_val, "cos"))
    print("[R148] best global jss:", formula_composition.global_metric(best_val, "jss"))
    print(best_val.to_string(index=False))

    if args.eval_test:
        best_test = formula_composition.eval_buckets(model, test_dl, device, "test")
        best_test.to_csv(out_dir / "r148_best_test.csv", index=False)

        print("\n===== R148 BEST TEST =====")
        print("[R148] best global cos:", formula_composition.global_metric(best_test, "cos"))
        print("[R148] best global jss:", formula_composition.global_metric(best_test, "jss"))
        print(best_test.to_string(index=False))

    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
