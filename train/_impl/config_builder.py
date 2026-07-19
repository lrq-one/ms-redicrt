#!/usr/bin/env python3
"""
Stage A: Unified V1 -> R119-style low-LR all-parameter transfer.

核心原则：
1. 仅使用当前本地仓库文件；
2. 不修改 diagnostics/101_unified_all_v1_v5_40ep.py；
3. 加载 V1 checkpoint 的模型权重，但不恢复历史 optimizer / epoch；
4. 保留 V1 网络结构和 V1 loss；
5. 只迁移 R119 的 optimizer、低学习率、scheduler、scope=all 和 no-r117 设置；
6. 根据 validation global cosine 保存 best checkpoint；
7. 训练结束后仅对 best validation checkpoint 测试一次。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML顶层必须是字典: {path}")
    return data


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path.resolve()

    attempted = "\n".join(f"  - {p}" for p in paths)
    raise FileNotFoundError(
        f"找不到{label}，已检查：\n{attempted}"
    )


def rotate_directory(path: Path) -> Path | None:
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak_{timestamp}")
    path.rename(backup)
    return backup


class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = (
            getattr(streams[0], "encoding", None)
            or "utf-8"
        )
        self.errors = (
            getattr(streams[0], "errors", None)
            or "strict"
        )

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def parse_v1_validation(
    v1_log: Path,
    output_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "log": str(v1_log),
        "found": False,
        "best": None,
        "last": None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not v1_log.is_file():
        text = (
            f"V1 validation log not found:\n"
            f"{v1_log}\n"
        )
        output_path.write_text(text, encoding="utf-8")
        return result

    log_text = v1_log.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    patterns = [
        re.compile(
            r"Epoch\s+(\d+),\s*step\s+(\d+)--\s*"
            r"(val_[^:\n]*cos_sim[^:\n]*):\s*"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        ),
        re.compile(
            r"(?:Epoch|epoch)[ =:]+(\d+).*?"
            r"(val_[^\s:=,]*cos_sim[^\s:=,]*).*?"
            r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        ),
    ]

    rows: list[dict[str, Any]] = []

    for pattern_index, pattern in enumerate(patterns):
        matches = list(pattern.finditer(log_text))
        if not matches:
            continue

        for match in matches:
            groups = match.groups()

            if pattern_index == 0:
                epoch = int(groups[0])
                step = int(groups[1])
                metric = groups[2]
                value = float(groups[3])
            else:
                epoch = int(groups[0])
                step = -1
                metric = groups[1]
                value = float(groups[2])

            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "metric": metric,
                    "value": value,
                }
            )

        if rows:
            break

    if rows:
        best = max(rows, key=lambda x: x["value"])
        last = max(rows, key=lambda x: (x["epoch"], x["step"]))

        result["found"] = True
        result["best"] = best
        result["last"] = last

        text = (
            f"V1 LOG: {v1_log}\n"
            f"V1 BEST VAL: {best}\n"
            f"V1 LAST VAL: {last}\n"
            f"NUM MATCHES: {len(rows)}\n"
        )
    else:
        text = (
            f"V1 LOG: {v1_log}\n"
            "V1 validation cosine not found by supported patterns\n"
        )

    output_path.write_text(text, encoding="utf-8")
    return result



def materialize_training_config(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)

    bundle = load_yaml(bundle_path)

    names = {
        "template": "template.yml",
        "base_stage": "base_stage.yml",
        "continuation_stage": "continuation_stage.yml",
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result: dict[str, Path] = {}

    for key, filename in names.items():
        value = bundle.get(key)

        if not isinstance(value, dict):
            raise KeyError(
                f"统一配置缺少字典节：{key}"
            )

        path = output_dir / filename

        path.write_text(
            yaml.safe_dump(
                value,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        result[key] = path

    return result

def prepare_effective_config(
    template_cfg: dict[str, Any],
    v1_custom: dict[str, Any],
    r119_custom: dict[str, Any],
    epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    V1 是结构和 loss 基础。

    这里只从 R119 迁移训练策略相关字段，绝不把 R119 的其他历史模块
    或结构开关覆盖进 V1。
    """
    effective_v1 = deep_merge(template_cfg, v1_custom)
    effective_r119 = deep_merge(template_cfg, r119_custom)

    transfer = copy.deepcopy(effective_v1)
    copied: dict[str, Any] = {}

    exact_training_keys = {
        "optimizer",
        "lr",
        "weight_decay",
        "lr_schedule",
        "lr_decay_rate",
        "lr_warmup_steps",
        "lr_decay_steps",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accumulate_grad_batches",
        "precision",
        "use_tensor_float32",
        "deterministic",
        "automatic_optimization",
    }

    for key in sorted(exact_training_keys):
        if key in effective_r119:
            transfer[key] = copy.deepcopy(effective_r119[key])
            copied[key] = copy.deepcopy(effective_r119[key])

    # 搜索本地配置中额外的 scheduler / optimizer 控制字段。
    # 只复制名称明确属于优化和学习率调度的字段。
    for key, value in effective_r119.items():
        key_lower = key.lower()

        is_scheduler_key = (
            "scheduler" in key_lower
            or key_lower.startswith("lr_")
            or key_lower.endswith("_lr")
            or key_lower.startswith("optimizer_")
        )

        if is_scheduler_key:
            transfer[key] = copy.deepcopy(value)
            copied[key] = copy.deepcopy(value)

    transfer["min_epochs"] = 1
    transfer["max_epochs"] = int(epochs)

    # 所有 V1 参数参与低学习率联合收敛。
    if "spectrum_refiner_train_scope" in transfer:
        transfer["spectrum_refiner_train_scope"] = "all"
        copied["spectrum_refiner_train_scope"] = "all"

    if "train_scope" in transfer:
        transfer["train_scope"] = "all"
        copied["train_scope"] = "all"

    # 明确关闭 R117。
    if "use_r117_support_oracle_reweight_loss" in transfer:
        transfer["use_r117_support_oracle_reweight_loss"] = False
        copied["use_r117_support_oracle_reweight_loss"] = False

    if "r117_support_oracle_weight" in transfer:
        transfer["r117_support_oracle_weight"] = 0.0
        copied["r117_support_oracle_weight"] = 0.0

    # 保证按照 validation cosine 选 best，训练结束后测试一次。
    transfer["eval_test_split"] = True
    transfer["disable_checkpoints"] = False
    transfer["delete_checkpoints"] = False
    transfer["upload_checkpoints"] = False
    transfer["checkpoint_save_last"] = True
    transfer["checkpoint_metric_mode"] = "max"

    checkpoint_metric = transfer.get(
        "checkpoint_metric",
        "val_cos_sim_0.01_epoch/mean",
    )

    if "cos_sim" not in str(checkpoint_metric):
        checkpoint_metric = "val_cos_sim_0.01_epoch/mean"

    transfer["checkpoint_metric"] = checkpoint_metric

    copied.update(
        {
            "min_epochs": transfer["min_epochs"],
            "max_epochs": transfer["max_epochs"],
            "eval_test_split": True,
            "checkpoint_metric": transfer["checkpoint_metric"],
            "checkpoint_metric_mode": "max",
            "checkpoint_save_last": True,
        }
    )

    # 禁止意外启用 R117。
    assert not bool(
        transfer.get(
            "use_r117_support_oracle_reweight_loss",
            False,
        )
    )
    assert float(
        transfer.get("r117_support_oracle_weight", 0.0)
    ) == 0.0

    return transfer, copied


def create_runtime_links(root: Path, run_dir: Path) -> None:
    """
    runner 在 run_dir 中执行，使 tmp_ckpt 不污染仓库根目录。
    同时给相对 data/... 路径创建本地链接。
    """
    link_names = [
        "data",
        "code",
        "config",
        "stages",    ]

    for name in link_names:
        target = root / name
        link = run_dir / name

        if not target.exists() or link.exists():
            continue

        link.symlink_to(
            target,
            target_is_directory=target.is_dir(),
        )


def install_local_pythonpath(root: Path) -> None:
    candidates = [
        root / "code" / "src",
        root / "code",
        root,
        root / "src",
    ]

    additions = [
        str(path)
        for path in candidates
        if path.exists()
    ]

    current = os.environ.get("PYTHONPATH", "")
    if current:
        additions.append(current)

    os.environ["PYTHONPATH"] = os.pathsep.join(additions)

    for path in reversed(additions):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="R119-style transfer epochs; first run should use 3",
    )
    parser.add_argument(
        "--run-name",
        default="v1_r119_reproduce",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="validation cosine early-stop patience",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1.0e-4,
        help="minimum validation cosine improvement",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"工作目录不存在: {root}")

    v1_cfg_path = root / (
        "runs/_config/base_stage.yml"
    )
    r119_cfg_path = root / (
        "runs/_config/continuation_stage.yml"
    )
    v1_ckpt_path = root / (
        "runs/v1/model_epoch39.ckpt"
    )
    v1_log_path = root / (
        "runs/v1/training.log"
    )

    template_path = first_existing(
        [
            root / "runs/_config/template.yml",
            root / "runs/_config/template.yml",
        ],
        "本地 template.yml",
    )

    required = {
        "V1配置": v1_cfg_path,
        "R119配置": r119_cfg_path,
        "V1 checkpoint": v1_ckpt_path,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "以下本地文件不存在：\n" + "\n".join(missing)
        )

    exact_val_path = (
        root
        / "artifacts"
        / "v1_exact_val.txt"
    )

    v1_val_result = parse_v1_validation(
        v1_log_path,
        exact_val_path,
    )

    run_dir = root / "runs" / args.run_name
    backup_dir = rotate_directory(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "training.log"
    log_file = log_path.open(
        "w",
        encoding="utf-8",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print("=" * 78)
        print("LOCAL V1 -> R119 TRANSFER")
        print("=" * 78)
        print(f"ROOT          : {root}")
        print(f"V1 CONFIG     : {v1_cfg_path}")
        print(f"R119 CONFIG   : {r119_cfg_path}")
        print(f"V1 CHECKPOINT : {v1_ckpt_path}")
        print(f"TEMPLATE      : {template_path}")
        print(f"RUN DIR       : {run_dir}")
        print(f"EPOCHS        : {args.epochs}")

        if backup_dir is not None:
            print(f"OLD RUN BACKUP: {backup_dir}")

        template_cfg = load_yaml(template_path)
        v1_custom = load_yaml(v1_cfg_path)
        r119_custom = load_yaml(r119_cfg_path)

        transfer_cfg, copied_training_settings = (
            prepare_effective_config(
                template_cfg=template_cfg,
                v1_custom=v1_custom,
                r119_custom=r119_custom,
                epochs=args.epochs,
            )
        )

        transfer_cfg["wandb_name"] = args.run_name
        transfer_cfg["wandb_group"] = (
            "MS2SPECTRA_V1_R119"
        )
        transfer_cfg["seed"] = 42

        # 早停由本脚本直接注入Trainer，避免依赖本地模板版本。
        if "early_stopping" in transfer_cfg:
            transfer_cfg["early_stopping"] = False

        resolved_cfg_path = (
            run_dir / "config.yml"
        )

        with resolved_cfg_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            yaml.safe_dump(
                transfer_cfg,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

        create_runtime_links(root, run_dir)
        install_local_pythonpath(root)

        manifest = {
            "stage": "A",
            "name": args.run_name,
            "root": str(root),
            "random_initialization": False,
            "weight_initialization": "Unified V1 epoch39",
            "optimizer_state_restored": False,
            "epoch_state_restored": False,
            "v1_config": str(v1_cfg_path),
            "r119_config": str(r119_cfg_path),
            "template": str(template_path),
            "v1_checkpoint": str(v1_ckpt_path),
            "v1_checkpoint_sha256": sha256_file(
                v1_ckpt_path
            ),
            "epochs": args.epochs,
            "early_stopping": {
                "enabled": True,
                "monitor": transfer_cfg["checkpoint_metric"],
                "mode": "max",
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "active_stage": "R119 low-LR transfer",
            },
            "copied_r119_training_settings": (
                copied_training_settings
            ),
            "v1_validation_parse": v1_val_result,
            "selection_metric": transfer_cfg[
                "checkpoint_metric"
            ],
            "test_policy": (
                "test once after fit using best validation checkpoint"
            ),
        }

        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n[R119 TRANSFER SETTINGS]")
        print(
            json.dumps(
                copied_training_settings,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\n[V1 VALIDATION AUDIT]")
        print(
            exact_val_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        )

        print("\n[LOAD LOCAL RUNNER]")

        import ms2spectra.workflow as runner

        try:
            from lightning.pytorch.callbacks import (
                EarlyStopping as LocalEarlyStopping,
            )
        except ModuleNotFoundError:
            from pytorch_lightning.callbacks import (
                EarlyStopping as LocalEarlyStopping,
            )

        original_trainer_cls = runner.pl.Trainer

        def trainer_with_local_early_stop(
            *trainer_args,
            callbacks=None,
            **trainer_kwargs,
        ):
            callback_list = list(callbacks or [])

            callback_list.append(
                LocalEarlyStopping(
                    monitor=transfer_cfg["checkpoint_metric"],
                    mode="max",
                    patience=int(args.patience),
                    min_delta=float(args.min_delta),
                    verbose=True,
                    strict=True,
                )
            )

            print(
                "[LOCAL EARLY STOP] "
                f"monitor={transfer_cfg['checkpoint_metric']}, "
                f"mode=max, "
                f"patience={args.patience}, "
                f"min_delta={args.min_delta}"
            )

            return original_trainer_cls(
                *trainer_args,
                callbacks=callback_list,
                **trainer_kwargs,
            )

        runner.pl.Trainer = trainer_with_local_early_stop

        if not hasattr(runner, "FragGNNPL"):
            raise RuntimeError(
                "本地 ms2spectra.workflow 中没有 FragGNNPL，"
                "拒绝猜测其他模型类。"
            )

        original_init = runner.FragGNNPL.__init__
        load_audit: dict[str, Any] = {}

        def patched_init(self, *model_args, **model_kwargs):
            original_init(
                self,
                *model_args,
                **model_kwargs,
            )

            print(
                "\n[V1 WEIGHT LOAD] "
                f"loading strictly from {v1_ckpt_path}"
            )

            checkpoint = torch.load(
                v1_ckpt_path,
                map_location="cpu",
            )

            if (
                isinstance(checkpoint, dict)
                and "state_dict" in checkpoint
            ):
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            if not isinstance(state_dict, dict):
                raise TypeError(
                    "V1 checkpoint 中没有可用的 state_dict"
                )

            # 严格加载，结构只要与 V1 不一致就立即报错，
            # 禁止 missing/unexpected key 静默继续训练。
            self.load_state_dict(
                state_dict,
                strict=True,
            )

            trainable = sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )
            total = sum(
                p.numel()
                for p in self.parameters()
            )

            load_audit.update(
                {
                    "strict": True,
                    "state_key_count": len(state_dict),
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                }
            )

            print(
                "[V1 WEIGHT LOAD] strict=True, "
                f"state_keys={len(state_dict)}, "
                f"trainable={trainable}, total={total}"
            )

        runner.FragGNNPL.__init__ = patched_init

        old_cwd = Path.cwd()
        os.chdir(run_dir)

        try:
            print("\n[START LOCAL TRAINING]")
            print(
                "注意：这是新的 optimizer 状态，"
                "不是从 V1 optimizer/epoch 恢复。"
            )

            runner.init_run(
                str(template_path),
                str(resolved_cfg_path),
                "disabled",
                None,
            )
        finally:
            os.chdir(old_cwd)
            runner.FragGNNPL.__init__ = original_init
            runner.pl.Trainer = original_trainer_cls

        tmp_ckpt_dir = run_dir / "tmp_ckpt"
        ckpt_files = sorted(
            tmp_ckpt_dir.glob("*.ckpt")
        )

        best_candidates = [
            path
            for path in ckpt_files
            if path.name != "last.ckpt"
        ]

        canonical_best = None
        if best_candidates:
            # ModelCheckpoint 默认只保留 validation 最优模型。
            source_best = max(
                best_candidates,
                key=lambda p: p.stat().st_mtime,
            )
            canonical_best = (
                run_dir / "model_best.ckpt"
            )
            shutil.copy2(
                source_best,
                canonical_best,
            )

            print(
                f"\n[BEST CHECKPOINT] {source_best}"
            )
            print(
                f"[CANONICAL COPY]  {canonical_best}"
            )

        last_ckpt = tmp_ckpt_dir / "last.ckpt"
        canonical_last = None

        if last_ckpt.is_file():
            canonical_last = (
                run_dir / "model_last.ckpt"
            )
            shutil.copy2(
                last_ckpt,
                canonical_last,
            )
            print(
                f"[LAST CHECKPOINT]  {canonical_last}"
            )

        manifest.update(
            {
                "weight_load_audit": load_audit,
                "completed": True,
                "checkpoint_files": [
                    str(p) for p in ckpt_files
                ],
                "canonical_best": (
                    str(canonical_best)
                    if canonical_best
                    else None
                ),
                "canonical_last": (
                    str(canonical_last)
                    if canonical_last
                    else None
                ),
            }
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("[SUCCESS] 阶段A本地迁移训练完成")
        print(f"日志     : {log_path}")
        print(f"配置     : {resolved_cfg_path}")
        print(f"manifest : {manifest_path}")
        print(f"V1审计   : {exact_val_path}")
        print("=" * 78)

        return 0

    except Exception:
        print("\n[FATAL] 阶段A失败，已停止，不会静默降级：")
        traceback.print_exc()

        failure_path = run_dir / "FAILED.txt"
        failure_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        return 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
