#!/usr/bin/env python3

from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from ms2spectra import workflow
from ms2spectra.training import FragGNNPL

from train._impl.trainer_helpers import (
    load_state_dict_any,
    make_trainer,
)


TEMPLATE = ROOT / "runs/_config/template.yml"

BASE_CONFIG = (
    ROOT
    / "runs"
    / "v2a_gine_cutchem_only"
    / "final"
    / "config.yml"
)

BASE_CHECKPOINT = (
    ROOT
    / "runs"
    / "v2a_gine_cutchem_only"
    / "final"
    / "model.ckpt"
)

OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "v2c_ce_trajectory_ablation"
)

MONITOR = "val_cos_sim_0.01_epoch/mean"
V2A_BASELINE = 0.5963571667671204


VARIANTS = {'control': {}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_full_config(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(
            f"配置顶层不是字典：{path}"
        )

    return config


def write_yaml(
    path: Path,
    config: dict[str, Any],
) -> None:
    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def rotate_output() -> None:
    if not OUTPUT_ROOT.exists():
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = OUTPUT_ROOT.with_name(
        OUTPUT_ROOT.name
        + ".bak_"
        + timestamp
    )

    OUTPUT_ROOT.rename(backup)

    print("[BACKUP]", backup)


def build_config(
    variant: str,
) -> dict[str, Any]:
    config = copy.deepcopy(
        load_full_config(
            BASE_CONFIG
        )
    )

    config.update(
        {
            "seed":
                42,

            "min_epochs":
                1,

            "max_epochs":
                5,

            "eval_test_split":
                False,

            "disable_checkpoints":
                False,

            "delete_checkpoints":
                False,

            "upload_checkpoints":
                False,

            "checkpoint_save_last":
                True,

            "checkpoint_metric":
                MONITOR,

            "checkpoint_metric_mode":
                "max",

            "num_sanity_val_steps":
                0,

            "compile":
                False,

            "wandb_name":
                f"V2C_{variant}",

            "wandb_group":
                "CONTROL_FINETUNING",

            

            

            

            "use_ce_depth_rank_loss":
                False,

            

            

            

            

            

            

            

            

            
        }
    )

    config.update(
        VARIANTS[variant]
    )

    # 必须保持V2A结构，不能混入连续CE。
    assert (
        config["frag_gnn_type"]
        == "GINE"
    )

    assert (
        config["frag_params"][
            "pyg_edge_feats"
        ]
        == ["cut_chem"]
    )

    assert (
        config["ce_insert_type"]
        == "embed"
    )




    assert not bool(
        config[
            "use_ce_depth_rank_loss"
        ]
    )

    return config


def train_variant(
    variant: str,
) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / variant

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = build_config(
        variant
    )

    config_path = (
        run_dir
        / "config.yml"
    )

    write_yaml(
        config_path,
        config,
    )

    print()
    print("=" * 96)
    print("CONTROL FINETUNING")
    print("=" * 96)

    print("variant           :", variant)
    print("base checkpoint   :", BASE_CHECKPOINT)
    print("output            :", run_dir)
    print("frag GNN          :", config["frag_gnn_type"])
    print(
        "frag edge feats   :",
        config["frag_params"]["pyg_edge_feats"],
    )
    print("CE encoder        :", config["ce_insert_type"])
    print("learning rate     :", config["lr"])
    print("weight decay      :", config["weight_decay"])
    print("test used         : False")
    print("=" * 96)

    pl.seed_everything(
        42,
        workers=True,
    )

    train_dataset, val_dataset = (
        workflow.init_dataset(
            config,
            splits=(
                "train",
                "val",
            ),
        )
    )

    train_loader = (
        workflow.init_dataloader(
            train_dataset,
            config,
        )
    )

    val_loader = (
        workflow.init_dataloader(
            val_dataset,
            config,
        )
    )

    model = FragGNNPL(
        **config
    )

    state_dict = load_state_dict_any(
        BASE_CHECKPOINT
    )

    incompatibility = (
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    )

    print(
        "[STRICT LOAD]",
        incompatibility,
    )

    trainer, checkpoint_callback = (
        make_trainer(
            run_dir=run_dir,
            config=config,
            patience=3,
            min_delta=1.0e-4,
            training=True,
        )
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if checkpoint_callback is None:
        raise RuntimeError(
            "checkpoint callback不存在"
        )

    best_source = Path(
        checkpoint_callback.best_model_path
    )

    if not best_source.is_file():
        raise FileNotFoundError(
            best_source
        )

    best_checkpoint = (
        run_dir
        / "model_best.ckpt"
    )

    shutil.copy2(
        best_source,
        best_checkpoint,
    )

    best_score = float(
        checkpoint_callback
        .best_model_score
        .detach()
        .cpu()
    )

    summary = {
        "variant":
            variant,

        "base_val_cosine":
            V2A_BASELINE,

        "best_val_cosine":
            best_score,

        "delta_vs_v2a":
            best_score
            - V2A_BASELINE,

        "checkpoint":
            str(best_checkpoint),

        "checkpoint_sha256":
            sha256_file(
                best_checkpoint
            ),

        

        

        "test_used":
            False,
    }

    (
        run_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"[{variant}] best validation cosine = "
        f"{best_score:.10f}"
    )

    del model
    del trainer
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    required = [
        TEMPLATE,
        BASE_CONFIG,
        BASE_CHECKPOINT,
        ROOT / "code/src/ms2spectra/training.py",
    ]

    missing = [
        path
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要文件：\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    rotate_output()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    control = train_variant(
        "control"
    )

    print()
    print("=" * 96)
    print("CONTROL FINETUNING COMPLETE")
    print("=" * 96)

    print(
        "best validation cosine:",
        f"{control['best_val_cosine']:.10f}",
    )

    print(
        "checkpoint:",
        control["checkpoint"],
    )

    print("test used: False")
    print("=" * 96)


if __name__ == "__main__":
    main()
