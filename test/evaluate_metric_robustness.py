#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

for import_path in (
    ROOT / "code/src",
    ROOT / "code",
    ROOT,
):
    sys.path.insert(0, str(import_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the locked R184B model with "
            "10 ppm Hungarian cosine similarity."
        )
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path(
            "runs/experiments/molecule_disjoint_3seeds/seed42"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--tolerance-floor-mz",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=2.0e-4,
    )
    return parser.parse_args()


def resolve_from_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label}不存在：{path}"
        )

    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def load_module(
    path: Path,
    module_name: str,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载模块：{path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chun_one_spectrum(
    true_mzs: torch.Tensor,
    true_ints: torch.Tensor,
    pred_mzs: torch.Tensor,
    pred_ints: torch.Tensor,
    ppm: float,
    tolerance_floor_mz: float,
) -> float:
    """
    CHUN definition:

    |m_true - m_pred|
        <= ppm * 1e-6 * max(m_true, tolerance_floor_mz)

    Intensities are L2-normalized over the full spectrum.
    Peak matching is one-to-one Hungarian assignment.
    No square-root intensity transform is applied.
    """
    if (
        true_mzs.numel() == 0
        or pred_mzs.numel() == 0
    ):
        return 0.0

    true_norm = true_ints / (
        torch.linalg.vector_norm(true_ints)
        .clamp_min(1.0e-12)
    )

    pred_norm = pred_ints / (
        torch.linalg.vector_norm(pred_ints)
        .clamp_min(1.0e-12)
    )

    tolerance_da = (
        float(ppm)
        * 1.0e-6
        * torch.clamp(
            true_mzs,
            min=float(tolerance_floor_mz),
        )
    )

    match_mask = (
        torch.abs(
            true_mzs.reshape(-1, 1)
            - pred_mzs.reshape(1, -1)
        )
        <= tolerance_da.reshape(-1, 1)
    )

    score_matrix = (
        match_mask.to(true_norm.dtype)
        * true_norm.reshape(-1, 1)
        * pred_norm.reshape(1, -1)
    )

    if not bool(match_mask.any()):
        return 0.0

    row_index, column_index = (
        linear_sum_assignment(
            score_matrix
            .detach()
            .cpu()
            .numpy(),
            maximize=True,
        )
    )

    if len(row_index) == 0:
        return 0.0

    row_index = torch.as_tensor(
        row_index,
        device=score_matrix.device,
        dtype=torch.long,
    )
    column_index = torch.as_tensor(
        column_index,
        device=score_matrix.device,
        dtype=torch.long,
    )

    return float(
        score_matrix[
            row_index,
            column_index,
        ]
        .sum()
        .detach()
        .cpu()
    )


def summarize(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []

    for bucket in (
        "global",
        "low_<=20",
        "mid_20_40",
        "high_>40",
    ):
        if bucket == "global":
            current = detail
        else:
            current = detail[
                detail["ce_bucket"].astype(str)
                == bucket
            ]

        if current.empty:
            continue

        output_rows.append(
            {
                "ce_bucket": bucket,
                "spec_count": int(len(current)),
                "mean_ce": float(
                    current["ce"].mean()
                ),
                "cos_0.01": float(
                    current["cos_0.01"].mean()
                ),
                "jss_0.01": float(
                    current["jss_0.01"].mean()
                ),
                "chun_10ppm": float(
                    current["chun_10ppm"].mean()
                ),
                "chun_std": float(
                    current["chun_10ppm"].std(
                        ddof=0
                    )
                ),
                "chun_median": float(
                    current["chun_10ppm"].median()
                ),
                "cos_raw_0.01": float(
                    current[
                        "cos_raw_0.01"
                    ].mean()
                ),
                "cos_sqrt_0.01": float(
                    current[
                        "cos_sqrt_0.01"
                    ].mean()
                ),
                "cos_raw_0.05": float(
                    current[
                        "cos_raw_0.05"
                    ].mean()
                ),
                "cos_sqrt_0.05": float(
                    current[
                        "cos_sqrt_0.05"
                    ].mean()
                ),
                "cos_raw_0.10": float(
                    current[
                        "cos_raw_0.10"
                    ].mean()
                ),
                "cos_sqrt_0.10": float(
                    current[
                        "cos_sqrt_0.10"
                    ].mean()
                ),
                "cos_raw_0.01_recomputed_audit": float(
                    current[
                        "cos_raw_0.01_recomputed_audit"
                    ].mean()
                ),
                "raw_0.01_parity_max_abs": float(
                    current[
                        "raw_0.01_parity_abs"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(output_rows)



def _metric_robustness_cosine_dense(
    true_dense: torch.Tensor,
    pred_dense: torch.Tensor,
) -> torch.Tensor:
    numerator = (
        true_dense
        * pred_dense
    ).sum(dim=1)

    true_norm = (
        true_dense
        .square()
        .sum(dim=1)
        .sqrt()
    )

    pred_norm = (
        pred_dense
        .square()
        .sum(dim=1)
        .sqrt()
    )

    denominator = (
        true_norm
        * pred_norm
    ).clamp_min(1.0e-12)

    return (
        numerator / denominator
    ).clamp(
        min=0.0,
        max=1.0,
    )



@torch.no_grad()
def evaluate_split(
    split: str,
    loader: Any,
    backbone: Any,
    allocator: Any,
    regressor: Any,
    extra_schema: Any,
    allocator_arguments: SimpleNamespace,
    candidate_reranker: Any,
    spectrum_allocator: Any,
    device: torch.device,
    ppm: float,
    tolerance_floor_mz: float,
    round_decimals: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ms2spectra.utils.spec_utils import (
        round_aggregate_peaks,
    )

    backbone.eval()
    allocator.eval()

    rows: list[dict[str, Any]] = []

    for batch in tqdm(
        loader,
        desc=f"CHUN {split}",
    ):
        batch = (
            candidate_reranker
            .move_to_device(
                batch,
                device,
            )
        )

        (
            result,
            features,
            lgbm_score,
            target_mass,
            _,
        ) = spectrum_allocator.build_batch_tensors(
            backbone,
            batch,
            candidate_reranker,
            regressor,
            extra_schema,
            allocator_arguments,
            split=split,
        )

        output = spectrum_allocator.forward_allocator(
            backbone,
            allocator,
            batch,
            result,
            features,
            lgbm_score,
            target_mass,
            candidate_reranker,
            allocator_arguments,
        )

        renderer_name = next(
            (
                name
                for name in (
                    "dense_by_round_bins_grad",
                    "dense_by_round_bins",
                )
                if callable(
                    getattr(
                        spectrum_allocator,
                        name,
                        None,
                    )
                )
            ),
            None,
        )

        if renderer_name is None:
            available = sorted(
                name
                for name in dir(
                    spectrum_allocator
                )
                if (
                    "dense" in name.lower()
                    or "bin" in name.lower()
                )
            )

            raise RuntimeError(
                "运行时spectrum_allocator中"
                "没有找到锁定dense renderer。"
                f"候选函数：{available}"
            )

        metric_renderer = getattr(
            spectrum_allocator,
            renderer_name,
        )

        batch_size = int(
            result["unique_id"].numel()
        )

        true_probability = (
            result["true_logprobs"]
            .exp()
            .float()
        )

        pred_probability = (
            output["new_logp"]
            .exp()
            .float()
        )

        true_batch_indices = (
            result["true_batch_idxs"]
            .long()
        )

        pred_batch_indices = (
            result["pred_batch_idxs"]
            .long()
        )

        mz_max = float(
            backbone.hparams.mz_max
        )

        locked_raw_cosine = (
            output["cos"]
            .detach()
            .float()
            .reshape(-1)
        )

        metric_raw_gpu = {}
        metric_sqrt_gpu = {}
        metric_recomputed_gpu = {}

        for (
            metric_label,
            metric_bin_res,
        ) in (
            ("0.01", 0.01),
            ("0.05", 0.05),
            ("0.10", 0.10),
        ):
            true_dense = metric_renderer(
                result["true_mzs"].float(),
                true_probability,
                true_batch_indices,
                batch_size=batch_size,
                mz_max=mz_max,
                bin_res=float(
                    metric_bin_res
                ),
            )

            pred_dense = metric_renderer(
                result["pred_mzs"].float(),
                pred_probability,
                pred_batch_indices,
                batch_size=batch_size,
                mz_max=mz_max,
                bin_res=float(
                    metric_bin_res
                ),
            )

            recomputed_raw = (
                _metric_robustness_cosine_dense(
                    true_dense,
                    pred_dense,
                )
            )

            sqrt_cosine = (
                _metric_robustness_cosine_dense(
                    true_dense
                    .clamp_min(0.0)
                    .sqrt(),
                    pred_dense
                    .clamp_min(0.0)
                    .sqrt(),
                )
            )

            metric_recomputed_gpu[
                metric_label
            ] = recomputed_raw

            metric_sqrt_gpu[
                metric_label
            ] = sqrt_cosine

            if metric_label == "0.01":
                metric_raw_gpu[
                    metric_label
                ] = locked_raw_cosine
            else:
                metric_raw_gpu[
                    metric_label
                ] = recomputed_raw

        raw_001_parity_abs = (
            metric_recomputed_gpu["0.01"]
            - locked_raw_cosine
        ).abs()

        metric_raw_cpu = {
            key: value.detach().cpu()
            for key, value
            in metric_raw_gpu.items()
        }

        metric_sqrt_cpu = {
            key: value.detach().cpu()
            for key, value
            in metric_sqrt_gpu.items()
        }

        recomputed_raw_001_cpu = (
            metric_recomputed_gpu["0.01"]
            .detach()
            .cpu()
        )

        raw_001_parity_abs_cpu = (
            raw_001_parity_abs
            .detach()
            .cpu()
        )

        true_mzs, true_ints, true_batch = (
            round_aggregate_peaks(
                result["true_mzs"].float(),
                result["true_logprobs"]
                .exp()
                .float(),
                result[
                    "true_batch_idxs"
                ].long(),
                decimals=int(
                    round_decimals
                ),
                agg="sum",
            )
        )

        pred_mzs, pred_ints, pred_batch = (
            round_aggregate_peaks(
                result["pred_mzs"].float(),
                output["new_logp"]
                .exp()
                .float(),
                result[
                    "pred_batch_idxs"
                ].long(),
                decimals=int(
                    round_decimals
                ),
                agg="sum",
            )
        )

        spectrum_ids = (
            result["unique_id"]
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()
            .astype(int)
        )

        collision_energy, _ = (
            candidate_reranker
            .find_ce(batch)
        )

        collision_energy = (
            collision_energy
            .detach()
            .cpu()
            .reshape(-1)
        )

        ce_buckets = (
            candidate_reranker
            .ce_bucket_names(
                collision_energy
            )
        )

        cosine = (
            output["cos"]
            .detach()
            .cpu()
            .reshape(-1)
        )

        jss = (
            output["jss"]
            .detach()
            .cpu()
            .reshape(-1)
        )

        for batch_index, spectrum_id in enumerate(
            spectrum_ids
        ):
            true_mask = (
                true_batch == batch_index
            )
            pred_mask = (
                pred_batch == batch_index
            )

            chun = chun_one_spectrum(
                true_mzs=true_mzs[true_mask],
                true_ints=true_ints[true_mask],
                pred_mzs=pred_mzs[pred_mask],
                pred_ints=pred_ints[pred_mask],
                ppm=ppm,
                tolerance_floor_mz=(
                    tolerance_floor_mz
                ),
            )

            rows.append(
                {
                    "split": split,
                    "spec_id": int(
                        spectrum_id
                    ),
                    "ce": float(
                        collision_energy[
                            batch_index
                        ]
                    ),
                    "ce_bucket": str(
                        ce_buckets[
                            batch_index
                        ]
                    ),
                    "cos_0.01": float(
                        cosine[batch_index]
                    ),
                    "jss_0.01": float(
                        jss[batch_index]
                    ),
                    "chun_10ppm": chun,
                    "cos_raw_0.01": float(
                        metric_raw_cpu[
                            "0.01"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.01": float(
                        metric_sqrt_cpu[
                            "0.01"
                        ][batch_index]
                    ),
                    "cos_raw_0.05": float(
                        metric_raw_cpu[
                            "0.05"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.05": float(
                        metric_sqrt_cpu[
                            "0.05"
                        ][batch_index]
                    ),
                    "cos_raw_0.10": float(
                        metric_raw_cpu[
                            "0.10"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.10": float(
                        metric_sqrt_cpu[
                            "0.10"
                        ][batch_index]
                    ),
                    "cos_raw_0.01_recomputed_audit": float(
                        recomputed_raw_001_cpu[
                            batch_index
                        ]
                    ),
                    "raw_0.01_parity_abs": float(
                        raw_001_parity_abs_cpu[
                            batch_index
                        ]
                    ),
                }
            )

    detail = pd.DataFrame(rows)

    if detail.empty:
        raise RuntimeError(
            f"{split}逐谱结果为空。"
        )

    maximum_renderer_parity_difference = float(
        detail[
            "raw_0.01_parity_abs"
        ].max()
    )

    mean_renderer_parity_difference = float(
        detail[
            "raw_0.01_parity_abs"
        ].mean()
    )

    print()
    print(
        f"[Metric robustness {split}] "
        f"renderer={renderer_name}"
    )
    print(
        f"[Metric robustness {split}] "
        "raw@0.01 parity: "
        f"max_abs="
        f"{maximum_renderer_parity_difference:.12e}, "
        f"mean_abs="
        f"{mean_renderer_parity_difference:.12e}"
    )

    if (
        maximum_renderer_parity_difference
        > 2.0e-6
    ):
        raise RuntimeError(
            f"{split} raw@0.01 renderer parity失败："
            f"max_abs="
            f"{maximum_renderer_parity_difference}"
        )

    metrics = summarize(detail)

    return metrics, detail


def main() -> None:
    args = parse_args()

    seed_dir = resolve_from_root(
        args.seed_dir
    )

    output_dir = (
        seed_dir / "metric_robustness"
        if args.output_dir is None
        else resolve_from_root(
            args.output_dir
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_path = require_file(
        ROOT / "runs/_config/template.yml",
        "模板配置",
    )

    def locate_seed_artifact(
        label: str,
        exact_candidates: list[Path],
        patterns: list[str],
        preferred_tokens: tuple[str, ...],
        fallback_candidates: list[Path] | None = None,
    ) -> Path:
        candidates: list[Path] = []

        for candidate in exact_candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                candidates.append(candidate)

        for pattern in patterns:
            for candidate in seed_dir.glob(pattern):
                candidate = candidate.resolve()
                if candidate.is_file():
                    candidates.append(candidate)

        unique_candidates: list[Path] = []
        seen: set[str] = set()

        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)

        if not unique_candidates:
            for candidate in fallback_candidates or []:
                candidate = candidate.resolve()
                if candidate.is_file():
                    unique_candidates.append(candidate)

        if not unique_candidates:
            print()
            print(f"[缺少文件] {label}")
            print("seed42目录：", seed_dir)
            print("当前seed42中的相关文件：")

            for candidate in sorted(
                seed_dir.rglob("*")
            ):
                if (
                    candidate.is_file()
                    and candidate.suffix
                    in {
                        ".yml",
                        ".yaml",
                        ".pt",
                        ".pkl",
                        ".ckpt",
                        ".json",
                    }
                ):
                    print("  ", candidate)

            raise FileNotFoundError(
                f"无法定位{label}"
            )

        def ranking_key(candidate: Path):
            path_text = str(candidate).lower()

            preferred_score = sum(
                token.lower() in path_text
                for token in preferred_tokens
            )

            seed_score = int(
                str(seed_dir) in str(candidate)
            )

            return (
                -seed_score,
                -preferred_score,
                len(candidate.parts),
                str(candidate),
            )

        selected = sorted(
            unique_candidates,
            key=ranking_key,
        )[0]

        print(f"[自动定位] {label}:")
        print(" ", selected)

        if len(unique_candidates) > 1:
            print(
                f"  共发现{len(unique_candidates)}个候选，"
                "已按seed42和路径关键词选择。"
            )

        return selected

    config_path = locate_seed_artifact(
        label="seed42模型配置",
        exact_candidates=[
            seed_dir
            / "v2c_ce_trajectory_ablation"
            / "control"
            / "config.yml",
        ],
        patterns=[
            "**/v2c_ce_trajectory_ablation/**/config.yml",
            "**/v2c_ce_trajectory_ablation/**/config.yaml",
            "**/control/config.yml",
            "**/control/config.yaml",
        ],
        preferred_tokens=(
            "v2c_ce_trajectory_ablation",
            "control",
        ),
        fallback_candidates=[
            ROOT
            / "runs"
            / "v2c_ce_trajectory_ablation"
            / "control"
            / "config.yml",
        ],
    )

    backbone_path = locate_seed_artifact(
        label="seed42 R160模型",
        exact_candidates=[
            seed_dir
            / "v2e_full_063"
            / "08_R160"
            / "r160_best_state.pt",
        ],
        patterns=[
            "**/r160_best_state.pt",
            "**/*r160*best*.pt",
        ],
        preferred_tokens=(
            "v2e_full_063",
            "08_r160",
            "r160_best_state",
        ),
    )

    reranker_path = locate_seed_artifact(
        label="seed42 R172D模型",
        exact_candidates=[
            seed_dir
            / "v2e_full_063"
            / "09_R172D"
            / "r170_regressor.pkl",
        ],
        patterns=[
            "**/r170_regressor.pkl",
            "**/*regressor*.pkl",
        ],
        preferred_tokens=(
            "v2e_full_063",
            "09_r172d",
            "r170_regressor",
        ),
    )

    allocator_path = locate_seed_artifact(
        label="seed42 R184B模型",
        exact_candidates=[
            seed_dir
            / "v2e_full_063"
            / "11_R184B"
            / "r184_allocator_best.pt",
        ],
        patterns=[
            "**/r184_allocator_best.pt",
            "**/*r184*allocator*best*.pt",
        ],
        preferred_tokens=(
            "v2e_full_063",
            "11_r184b",
            "r184_allocator_best",
        ),
    )

    reranker_script = require_file(
        ROOT
        / "train/_impl/refinement_steps"
        / "candidate_reranker.py",
        "candidate_reranker.py",
    )

    allocator_script = require_file(
        ROOT
        / "train/_impl/refinement_steps"
        / "spectrum_allocator.py",
        "spectrum_allocator.py",
    )

    seed_everything(
        int(args.seed)
    )

    candidate_reranker = load_module(
        reranker_script,
        "seed42_candidate_reranker",
    )

    spectrum_allocator = load_module(
        allocator_script,
        "seed42_spectrum_allocator",
    )

    def lgbm_predict_with_names(
        regressor: Any,
        features: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        score_clip: float,
    ) -> torch.Tensor:
        feature_array = (
            features
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        feature_names = getattr(
            regressor,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            feature_names = getattr(
                regressor,
                "feature_name_",
                None,
            )

        predict_input: Any = feature_array

        if feature_names is not None:
            names = [
                str(name)
                for name in list(
                    feature_names
                )
            ]

            if len(names) == int(
                feature_array.shape[1]
            ):
                predict_input = pd.DataFrame(
                    feature_array,
                    columns=names,
                )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"X does not have valid "
                    r"feature names.*"
                ),
                category=UserWarning,
            )

            score = (
                regressor
                .predict(predict_input)
                .astype(np.float32)
            )

        score = np.clip(
            score,
            -float(score_clip),
            float(score_clip),
        )

        return torch.from_numpy(
            score
        ).to(
            device=device,
            dtype=dtype,
        )

    spectrum_allocator.lgbm_predict = (
        lgbm_predict_with_names
    )

    from ms2spectra.training import (
        FragGNNPL,
    )
    from ms2spectra.workflow import (
        init_dataloader,
        init_dataset,
        load_config,
    )

    with reranker_path.open(
        "rb"
    ) as handle:
        reranker_package = pickle.load(
            handle
        )

    regressor = reranker_package["model"]

    allocator_package = torch.load(
        allocator_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_arguments = dict(
        allocator_package["args"]
    )

    allocator_arguments = SimpleNamespace(
        **saved_arguments
    )

    extra_schema = allocator_package.get(
        "extra_schema",
        reranker_package.get(
            "extra_schema",
            [],
        ),
    )

    saved_validation_cosine = float(
        allocator_package.get(
            "best_val_cos",
            float("nan"),
        )
    )

    config = load_config(
        template_path,
        config_path,
    )

    config = (
        candidate_reranker
        .force_r160_arch(config)
    )

    validation_dataset, test_dataset = (
        init_dataset(
            config,
            splits=("val", "test"),
        )
    )

    validation_loader = init_dataloader(
        validation_dataset,
        config,
    )

    test_loader = init_dataloader(
        test_dataset,
        config,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    backbone = FragGNNPL(
        **config
    )

    backbone_state = (
        candidate_reranker
        .load_state_dict_any(
            backbone_path
        )
    )

    missing, unexpected = (
        backbone.load_state_dict(
            backbone_state,
            strict=False,
        )
    )

    print(
        "Backbone missing keys:",
        len(missing),
    )
    print(
        "Backbone unexpected keys:",
        len(unexpected),
    )

    backbone = backbone.to(device)
    backbone.eval()

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    allocator = (
        spectrum_allocator
        .ResidualAllocator(
            input_dim=int(
                allocator_package[
                    "input_dim"
                ]
            ),
            hidden=int(
                saved_arguments["hidden"]
            ),
            layers=int(
                saved_arguments["layers"]
            ),
            dropout=float(
                saved_arguments["dropout"]
            ),
            score_clip=float(
                saved_arguments[
                    "score_clip"
                ]
            ),
        )
        .to(device)
    )

    allocator.load_state_dict(
        allocator_package["model"]
    )
    allocator.eval()

    print()
    print("=" * 88)
    print(
        "LOCKED METRIC ROBUSTNESS EVALUATION"
    )
    print("=" * 88)
    print("seed_dir:", seed_dir)
    print("output_dir:", output_dir)
    print("ppm:", float(args.ppm))
    print(
        "tolerance floor:",
        float(
            args.tolerance_floor_mz
        ),
    )
    print(
        "round decimals:",
        int(args.round_decimals),
    )
    print("intensity transform: none")
    print("precursor peak: kept")
    print("matching: Hungarian one-to-one")
    print("retraining: false")
    print("=" * 88)

    validation_metrics, validation_detail = (
        evaluate_split(
            split="val",
            loader=validation_loader,
            backbone=backbone,
            allocator=allocator,
            regressor=regressor,
            extra_schema=extra_schema,
            allocator_arguments=(
                allocator_arguments
            ),
            candidate_reranker=(
                candidate_reranker
            ),
            spectrum_allocator=(
                spectrum_allocator
            ),
            device=device,
            ppm=float(args.ppm),
            tolerance_floor_mz=float(
                args.tolerance_floor_mz
            ),
            round_decimals=int(
                args.round_decimals
            ),
        )
    )

    validation_metrics.to_csv(
        output_dir
        / "validation_metrics.csv",
        index=False,
    )

    validation_detail.to_csv(
        output_dir
        / "validation_per_spectrum_metrics.csv",
        index=False,
    )

    validation_global = (
        validation_metrics[
            validation_metrics[
                "ce_bucket"
            ].astype(str)
            == "global"
        ]
        .iloc[0]
    )

    recomputed_validation_cosine = float(
        validation_global[
            "cos_0.01"
        ]
    )

    parity_difference = (
        recomputed_validation_cosine
        - saved_validation_cosine
    )

    parity_passed = (
        not math.isfinite(
            saved_validation_cosine
        )
        or abs(parity_difference)
        <= float(
            args.parity_tolerance
        )
    )

    print()
    print("VALIDATION RESULT")
    print(
        validation_metrics.to_string(
            index=False
        )
    )
    print()
    print(
        "Saved validation cosine:",
        saved_validation_cosine,
    )
    print(
        "Recomputed validation cosine:",
        recomputed_validation_cosine,
    )
    print(
        "Parity difference:",
        parity_difference,
    )
    print(
        "Parity passed:",
        parity_passed,
    )

    if not parity_passed:
        raise RuntimeError(
            "验证集0.01 Da cosine复现失败，"
            "为防止评错模型，停止test评价。"
        )

    test_metrics, test_detail = (
        evaluate_split(
            split="test",
            loader=test_loader,
            backbone=backbone,
            allocator=allocator,
            regressor=regressor,
            extra_schema=extra_schema,
            allocator_arguments=(
                allocator_arguments
            ),
            candidate_reranker=(
                candidate_reranker
            ),
            spectrum_allocator=(
                spectrum_allocator
            ),
            device=device,
            ppm=float(args.ppm),
            tolerance_floor_mz=float(
                args.tolerance_floor_mz
            ),
            round_decimals=int(
                args.round_decimals
            ),
        )
    )

    test_metrics.to_csv(
        output_dir
        / "test_metrics.csv",
        index=False,
    )

    test_detail.to_csv(
        output_dir
        / "test_per_spectrum_metrics.csv",
        index=False,
    )

    result = {
        "experiment": (
            "locked_seed42_chun_10ppm"
        ),
        "retrained": False,
        "model_selection_changed": False,
        "seed_dir": str(seed_dir),
        "metric": {
            "name": "CHUN",
            "ppm": float(args.ppm),
            "tolerance_rule": (
                "|m_true-m_pred| <= "
                "ppm*1e-6*max(m_true,200)"
            ),
            "tolerance_floor_mz": float(
                args.tolerance_floor_mz
            ),
            "round_decimals": int(
                args.round_decimals
            ),
            "intensity_transform": "none",
            "normalization": (
                "per-spectrum L2"
            ),
            "matching": (
                "Hungarian one-to-one"
            ),
            "precursor_peak": "kept",
        },
        "validation_parity": {
            "saved_cos_0.01": (
                saved_validation_cosine
            ),
            "recomputed_cos_0.01": (
                recomputed_validation_cosine
            ),
            "difference": (
                parity_difference
            ),
            "tolerance": float(
                args.parity_tolerance
            ),
            "passed": bool(
                parity_passed
            ),
        },
        "validation": json.loads(
            validation_metrics.to_json(
                orient="records"
            )
        ),
        "test": json.loads(
            test_metrics.to_json(
                orient="records"
            )
        ),
        "artifacts": {
            "config": str(config_path),
            "backbone": str(
                backbone_path
            ),
            "reranker": str(
                reranker_path
            ),
            "allocator": str(
                allocator_path
            ),
        },
    }

    result_path = (
        output_dir
        / "metric_robustness_result.json"
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
    print("TEST RESULT")
    print(
        test_metrics.to_string(
            index=False
        )
    )
    print()
    print("WROTE:", result_path)


if __name__ == "__main__":
    main()
