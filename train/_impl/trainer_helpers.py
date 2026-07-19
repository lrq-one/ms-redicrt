#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
    )
except ModuleNotFoundError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
    )


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "code/src"),
)

sys.path.insert(
    0,
    str(ROOT / "code"),
)

sys.path.insert(
    0,
    str(ROOT),
)

from ms2spectra import workflow
from ms2spectra.training import FragGNNPL


TEMPLATE = ROOT / "runs/_config/template.yml"

BASE_CONFIG = (
    ROOT
    / "runs"
    / "from_scratch_v1_40_r119_10_seed42"
    / "final"
    / "config.yml"
)

BASE_CHECKPOINT = (
    ROOT
    / "runs"
    / "from_scratch_v1_40_r119_10_seed42"
    / "final"
    / "model.ckpt"
)

OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "r121_allocation_ablation"
)

MONITOR = "val_cos_sim_0.01_epoch/mean"


VARIANTS = {
    "control": {
        "use_r117_support_oracle_reweight_loss":
            False,

        "r117_support_oracle_weight":
            0.0,

        "r117_false_mass_weight":
            0.0,

        "r117_min_covered_true_mass":
            0.75,
    },

    "alloc005": {
        "use_r117_support_oracle_reweight_loss":
            True,

        "r117_support_oracle_weight":
            0.005,

        "r117_false_mass_weight":
            0.10,

        "r117_min_covered_true_mass":
            0.75,
    },

    "alloc010": {
        "use_r117_support_oracle_reweight_loss":
            True,

        "r117_support_oracle_weight":
            0.010,

        "r117_false_mass_weight":
            0.20,

        "r117_min_covered_true_mass":
            0.75,
    },
}


def sha256_file(
    path: Path,
    block_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_state_dict_any(
    path: Path,
) -> dict[str, torch.Tensor]:
    pack = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(pack, dict)
        and "state_dict" in pack
    ):
        return pack["state_dict"]

    if isinstance(pack, dict):
        return pack

    raise TypeError(
        f"无法从checkpoint读取state_dict：{path}"
    )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(
                value.detach().cpu()
            )

        return (
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(value)


def verify_inputs() -> None:
    required = [
        TEMPLATE,
        BASE_CONFIG,
        BASE_CHECKPOINT,
        ROOT
        / "code/src/ms2spectra/training.py",
    ]

    missing = [
        path
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要输入：\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    training_source = (
        ROOT
        / "code/src/ms2spectra/training.py"
    ).read_text(
        encoding="utf-8",
        errors="strict",
    )

    markers = [
        "_r117_support_oracle_reweight_loss",
        "use_r117_support_oracle_reweight_loss",
        "train_r117_support_oracle_loss",
        "_r98_apply_binned_spectrum_renderer",
    ]

    absent = [
        marker
        for marker in markers
        if marker not in training_source
    ]

    if absent:
        raise RuntimeError(
            "当前training.py缺少必要功能："
            + ", ".join(absent)
        )


def build_config(
    variant: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    loaded = workflow.load_config(
        str(TEMPLATE),
        str(BASE_CONFIG),
    )

    config = copy.deepcopy(
        dict(loaded)
    )

    config.update(
        {
            "seed":
                42,

            "lr":
                float(learning_rate),

            "weight_decay":
                float(weight_decay),

            "min_epochs":
                1,

            "max_epochs":
                int(epochs),

            "checkpoint_metric":
                MONITOR,

            "checkpoint_metric_mode":
                "max",

            "checkpoint_save_last":
                True,

            "disable_checkpoints":
                False,

            "delete_checkpoints":
                False,

            "eval_test_split":
                False,

            "compile":
                False,

            "num_sanity_val_steps":
                0,

            "use_tensor_float32":
                True,

            "spectrum_refiner_train_scope":
                "all",

            "train_scope":
                "all",

            "use_binned_spectrum_renderer":
                True,

            "binned_spectrum_renderer_apply_train":
                True,

            "binned_spectrum_renderer_bin_res":
                0.01,

            "r117_support_oracle_every_n_steps":
                1,

            "r117_oracle_bin_res":
                0.01,

            "r117_eps":
                1.0e-12,

            "wandb_name":
                f"R121_{variant}",

            "wandb_group":
                "R121_CURRENT_0627_ALLOCATION_ABLATION",
        }
    )

    config.update(
        VARIANTS[variant]
    )

    return config


def make_trainer(
    run_dir: Path,
    config: dict[str, Any],
    patience: int,
    min_delta: float,
    training: bool,
) -> tuple[
    pl.Trainer,
    ModelCheckpoint | None,
]:
    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )

    common_arguments = {
        "accelerator":
            accelerator,

        "devices":
            1,

        "precision":
            32,

        "logger":
            False,

        "enable_model_summary":
            False,

        "enable_progress_bar":
            True,

        "log_every_n_steps":
            int(
                config.get(
                    "log_every_n_steps",
                    100,
                )
            ),

        "num_sanity_val_steps":
            0,

        "gradient_clip_val":
            float(
                config.get(
                    "gradient_clip_val",
                    0.5,
                )
            ),

        "accumulate_grad_batches":
            int(
                config.get(
                    "accumulate_grad_batches",
                    1,
                )
            ),

        "deterministic":
            False,
    }

    if not training:
        trainer = pl.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            **common_arguments,
        )

        return trainer, None

    checkpoint_dir = (
        run_dir / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch-{epoch:02d}",
        monitor=MONITOR,
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor=MONITOR,
        mode="max",
        patience=int(patience),
        min_delta=float(min_delta),
        check_finite=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=int(
            config["max_epochs"]
        ),
        callbacks=[
            checkpoint_callback,
            early_stopping,
        ],
        enable_checkpointing=True,
        **common_arguments,
    )

    return trainer, checkpoint_callback


def train_variant(
    variant: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    overwrite: bool,
) -> None:
    run_dir = OUTPUT_ROOT / variant

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出目录已存在：{run_dir}\n"
                "需要重跑时增加 --overwrite"
            )

        shutil.rmtree(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pl.seed_everything(
        42,
        workers=True,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass

    config = build_config(
        variant=variant,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    config_path = run_dir / "config.yml"

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("R121 ALLOCATION ABLATION")
    print("=" * 88)
    print("variant        :", variant)
    print("base config    :", BASE_CONFIG)
    print("base checkpoint:", BASE_CHECKPOINT)
    print("output         :", run_dir)
    print("epochs         :", epochs)
    print("lr             :", learning_rate)
    print(
        "R117 enabled   :",
        config[
            "use_r117_support_oracle_reweight_loss"
        ],
    )
    print(
        "R117 weight    :",
        config[
            "r117_support_oracle_weight"
        ],
    )
    print(
        "false weight   :",
        config[
            "r117_false_mass_weight"
        ],
    )
    print(
        "min coverage   :",
        config[
            "r117_min_covered_true_mass"
        ],
    )
    print("=" * 88)

    train_dataset, val_dataset = (
        workflow.init_dataset(
            config,
            splits=("train", "val"),
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

    model = FragGNNPL(**config)

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

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[PARAMETERS] trainable={trainable}, "
        f"total={total}"
    )

    trainer, checkpoint_callback = (
        make_trainer(
            run_dir=run_dir,
            config=config,
            patience=patience,
            min_delta=min_delta,
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
            "checkpoint callback未创建"
        )

    best_path = Path(
        checkpoint_callback.best_model_path
    )

    if not best_path.is_file():
        raise FileNotFoundError(
            f"最佳checkpoint不存在：{best_path}"
        )

    canonical_best = (
        run_dir / "model_best.ckpt"
    )

    shutil.copy2(
        best_path,
        canonical_best,
    )

    last_source = (
        run_dir
        / "checkpoints"
        / "last.ckpt"
    )

    canonical_last = (
        run_dir / "model_last.ckpt"
    )

    if last_source.is_file():
        shutil.copy2(
            last_source,
            canonical_last,
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

        "selection_split":
            "validation",

        "monitor":
            MONITOR,

        "best_val_cosine":
            best_score,

        "best_checkpoint":
            str(canonical_best),

        "source_best_checkpoint":
            str(best_path),

        "base_checkpoint":
            str(BASE_CHECKPOINT),

        "base_checkpoint_sha256":
            sha256_file(
                BASE_CHECKPOINT
            ),

        "best_checkpoint_sha256":
            sha256_file(
                canonical_best
            ),

        "epochs_requested":
            int(epochs),

        "lr":
            float(learning_rate),

        "weight_decay":
            float(weight_decay),

        **VARIANTS[variant],
    }

    (
        run_dir / "screen_summary.json"
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
    print("=" * 88)
    print("SCREENING COMPLETE")
    print("=" * 88)
    print(
        "variant       :",
        variant,
    )
    print(
        "best val cos  :",
        f"{best_score:.9f}",
    )
    print(
        "best checkpoint:",
        canonical_best,
    )
    print("=" * 88)


def make_leaderboard() -> None:
    rows = []

    for variant in VARIANTS:
        summary_path = (
            OUTPUT_ROOT
            / variant
            / "screen_summary.json"
        )

        if not summary_path.is_file():
            continue

        row = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "没有找到任何screen_summary.json"
        )

    rows.sort(
        key=lambda row: float(
            row["best_val_cosine"]
        ),
        reverse=True,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    leaderboard_path = (
        OUTPUT_ROOT / "leaderboard.csv"
    )

    fields = [
        "rank",
        "variant",
        "best_val_cosine",
        "use_r117_support_oracle_reweight_loss",
        "r117_support_oracle_weight",
        "r117_false_mass_weight",
        "r117_min_covered_true_mass",
        "best_checkpoint",
    ]

    with leaderboard_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for rank, row in enumerate(
            rows,
            start=1,
        ):
            writer.writerow(
                {
                    "rank":
                        rank,

                    "variant":
                        row["variant"],

                    "best_val_cosine":
                        row[
                            "best_val_cosine"
                        ],

                    "use_r117_support_oracle_reweight_loss":
                        row[
                            "use_r117_support_oracle_reweight_loss"
                        ],

                    "r117_support_oracle_weight":
                        row[
                            "r117_support_oracle_weight"
                        ],

                    "r117_false_mass_weight":
                        row[
                            "r117_false_mass_weight"
                        ],

                    "r117_min_covered_true_mass":
                        row[
                            "r117_min_covered_true_mass"
                        ],

                    "best_checkpoint":
                        row[
                            "best_checkpoint"
                        ],
                }
            )

    selected = str(
        rows[0]["variant"]
    )

    (
        OUTPUT_ROOT
        / "selected_variant.txt"
    ).write_text(
        selected + "\n",
        encoding="utf-8",
    )

    (
        OUTPUT_ROOT
        / "selected_summary.json"
    ).write_text(
        json.dumps(
            rows[0],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("VALIDATION LEADERBOARD")
    print("=" * 88)

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{row['variant']:<10} "
            f"val_cos="
            f"{float(row['best_val_cosine']):.9f}"
        )

    print()
    print(
        "[SELECTED BY VALIDATION]",
        selected,
    )
    print(
        "[LEADERBOARD]",
        leaderboard_path,
    )
    print("=" * 88)


def test_variant(
    variant: str,
) -> None:
    run_dir = OUTPUT_ROOT / variant

    config_path = run_dir / "config.yml"
    checkpoint_path = (
        run_dir / "model_best.ckpt"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    config = workflow.load_config(
        str(TEMPLATE),
        str(config_path),
    )

    test_dataset, = (
        workflow.init_dataset(
            config,
            splits=("test",),
        )
    )

    test_loader = (
        workflow.init_dataloader(
            test_dataset,
            config,
        )
    )

    model = FragGNNPL(**config)

    state_dict = load_state_dict_any(
        checkpoint_path
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    trainer, _ = make_trainer(
        run_dir=run_dir,
        config=config,
        patience=1,
        min_delta=0.0,
        training=False,
    )

    test_output = trainer.test(
        model,
        dataloaders=test_loader,
        verbose=True,
    )

    result = {
        "variant":
            variant,

        "selection_rule":
            "highest validation cosine; test evaluated once",

        "checkpoint":
            str(checkpoint_path),

        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),

        "test_output":
            json_safe(test_output),
    }

    result_path = (
        run_dir / "test_result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("SELECTED VARIANT TEST COMPLETE")
    print("=" * 88)
    print("variant:", variant)
    print("result :", result_path)
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "leaderboard",
            "test",
        ],
        required=True,
    )

    parser.add_argument(
        "--variant",
        choices=list(VARIANTS),
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2.0e-6,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0e-5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1.0e-4,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    verify_inputs()

    arguments = parse_args()

    if arguments.mode == "leaderboard":
        make_leaderboard()
        return

    if arguments.variant is None:
        raise ValueError(
            "--mode train/test时必须提供--variant"
        )

    if arguments.mode == "train":
        train_variant(
            variant=arguments.variant,
            epochs=arguments.epochs,
            learning_rate=arguments.lr,
            weight_decay=arguments.weight_decay,
            patience=arguments.patience,
            min_delta=arguments.min_delta,
            overwrite=arguments.overwrite,
        )

        return

    if arguments.mode == "test":
        test_variant(
            arguments.variant
        )

        return

    raise RuntimeError(
        arguments.mode
    )


if __name__ == "__main__":
    main()
