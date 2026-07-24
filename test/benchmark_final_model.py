from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path.cwd()

EVALUATOR_PATH = (
    ROOT
    / "test"
    / "evaluate_chun_10ppm.py"
)

PROCESS_START_TIME = time.perf_counter()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def option_value(
    arguments: list[str],
    option: str,
) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise RuntimeError(
                    f"{option}后面缺少参数值。"
                )

            return arguments[index + 1]

        prefix = option + "="

        if argument.startswith(prefix):
            return argument[len(prefix):]

    return None


def synchronize(
    device: torch.device,
) -> None:
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)


def parameter_statistics(
    module: Any,
) -> dict[str, int]:
    if not hasattr(module, "parameters"):
        return {
            "total": 0,
            "trainable": 0,
        }

    parameters = list(
        module.parameters()
    )

    return {
        "total": int(
            sum(
                parameter.numel()
                for parameter
                in parameters
            )
        ),
        "trainable": int(
            sum(
                parameter.numel()
                for parameter
                in parameters
                if parameter.requires_grad
            )
        ),
    }


def regressor_statistics(
    regressor: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": (
            f"{type(regressor).__module__}."
            f"{type(regressor).__name__}"
        ),
    }

    for attribute in (
        "n_estimators_",
        "n_features_in_",
        "best_iteration_",
    ):
        if hasattr(regressor, attribute):
            value = getattr(
                regressor,
                attribute,
            )

            if isinstance(
                value,
                (int, float, str, bool),
            ):
                result[attribute] = value

    booster = getattr(
        regressor,
        "booster_",
        None,
    )

    if booster is not None:
        try:
            result["num_trees"] = int(
                booster.num_trees()
            )
        except Exception:
            pass

    try:
        if (
            "num_trees" not in result
            and hasattr(
                regressor,
                "num_trees",
            )
        ):
            result["num_trees"] = int(
                regressor.num_trees()
            )
    except Exception:
        pass

    return result


def artifact_inventory(
    seed_dir: Path,
) -> list[dict[str, Any]]:
    expected_names = (
        "r160_best_state.pt",
        "r170_regressor.pkl",
        "r184_allocator_best.pt",
        "config.yml",
    )

    records = []

    for filename in expected_names:
        matches = sorted(
            seed_dir.rglob(filename)
        )

        for path in matches:
            records.append(
                {
                    "name": filename,
                    "path": str(
                        path.resolve()
                    ),
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "size_mb": float(
                        path.stat().st_size
                        / 1024**2
                    ),
                }
            )

    return records


def load_evaluator(
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(path)

    specification = (
        importlib.util
        .spec_from_file_location(
            "locked_efficiency_evaluator",
            str(path),
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"无法加载：{path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    return module


benchmark_parser = argparse.ArgumentParser(
    add_help=False,
)

benchmark_parser.add_argument(
    "--warmup-batches",
    type=int,
    default=10,
)

benchmark_arguments, evaluator_arguments = (
    benchmark_parser.parse_known_args()
)

seed_dir_argument = option_value(
    evaluator_arguments,
    "--seed-dir",
)

output_dir_argument = option_value(
    evaluator_arguments,
    "--output-dir",
)

if seed_dir_argument is None:
    raise RuntimeError(
        "必须提供--seed-dir。"
    )

if output_dir_argument is None:
    raise RuntimeError(
        "必须提供--output-dir。"
    )

seed_dir = resolve_path(
    seed_dir_argument
)

output_dir = resolve_path(
    output_dir_argument
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path = (
    output_dir
    / "efficiency_benchmark.json"
)

batch_path = (
    output_dir
    / "efficiency_per_batch.csv"
)

evaluator = load_evaluator(
    EVALUATOR_PATH
)

original_evaluate_split = (
    evaluator.evaluate_split
)

benchmark_state: dict[str, Any] = {}


def timed_evaluate_split(
    split: str,
    loader: Any,
    backbone: Any,
    allocator: Any,
    regressor: Any,
    extra_schema: Any,
    allocator_arguments: Any,
    candidate_reranker: Any,
    spectrum_allocator: Any,
    device: torch.device,
    ppm: float,
    tolerance_floor_mz: float,
    round_decimals: int,
):
    if split != "test":
        return original_evaluate_split(
            split,
            loader,
            backbone,
            allocator,
            regressor,
            extra_schema,
            allocator_arguments,
            candidate_reranker,
            spectrum_allocator,
            device,
            ppm,
            tolerance_floor_mz,
            round_decimals,
        )

    device = torch.device(device)

    original_build_batch_tensors = (
        spectrum_allocator
        .build_batch_tensors
    )

    original_forward_allocator = (
        spectrum_allocator
        .forward_allocator
    )

    build_times: list[float] = []
    forward_times: list[float] = []
    batch_sizes: list[int] = []

    def timed_build_batch_tensors(
        *args: Any,
        **kwargs: Any,
    ):
        synchronize(device)
        start = time.perf_counter()

        result = (
            original_build_batch_tensors(
                *args,
                **kwargs,
            )
        )

        synchronize(device)
        elapsed = (
            time.perf_counter()
            - start
        )

        build_times.append(
            float(elapsed)
        )

        try:
            result_dictionary = result[0]

            batch_size = int(
                result_dictionary[
                    "unique_id"
                ].numel()
            )
        except Exception as error:
            raise RuntimeError(
                "无法从build_batch_tensors"
                "输出中取得batch size。"
            ) from error

        batch_sizes.append(
            batch_size
        )

        return result

    def timed_forward_allocator(
        *args: Any,
        **kwargs: Any,
    ):
        synchronize(device)
        start = time.perf_counter()

        result = (
            original_forward_allocator(
                *args,
                **kwargs,
            )
        )

        synchronize(device)
        elapsed = (
            time.perf_counter()
            - start
        )

        forward_times.append(
            float(elapsed)
        )

        return result

    spectrum_allocator.build_batch_tensors = (
        timed_build_batch_tensors
    )

    spectrum_allocator.forward_allocator = (
        timed_forward_allocator
    )

    if (
        device.type == "cuda"
        and torch.cuda.is_available()
    ):
        synchronize(device)

        torch.cuda.reset_peak_memory_stats(
            device
        )

        baseline_allocated_bytes = int(
            torch.cuda.memory_allocated(
                device
            )
        )

        baseline_reserved_bytes = int(
            torch.cuda.memory_reserved(
                device
            )
        )
    else:
        baseline_allocated_bytes = 0
        baseline_reserved_bytes = 0

    evaluation_start = time.perf_counter()

    try:
        metrics, detail = (
            original_evaluate_split(
                split,
                loader,
                backbone,
                allocator,
                regressor,
                extra_schema,
                allocator_arguments,
                candidate_reranker,
                spectrum_allocator,
                device,
                ppm,
                tolerance_floor_mz,
                round_decimals,
            )
        )
    finally:
        synchronize(device)

        test_evaluation_seconds = float(
            time.perf_counter()
            - evaluation_start
        )

        spectrum_allocator.build_batch_tensors = (
            original_build_batch_tensors
        )

        spectrum_allocator.forward_allocator = (
            original_forward_allocator
        )

    if not (
        len(build_times)
        == len(forward_times)
        == len(batch_sizes)
    ):
        raise RuntimeError(
            "计时调用数量不一致："
            f"build={len(build_times)}, "
            f"forward={len(forward_times)}, "
            f"batch={len(batch_sizes)}"
        )

    if len(batch_sizes) == 0:
        raise RuntimeError(
            "测试集没有产生任何batch。"
        )

    warmup_batches = min(
        max(
            0,
            int(
                benchmark_arguments
                .warmup_batches
            ),
        ),
        max(
            0,
            len(batch_sizes) - 1,
        ),
    )

    records = []

    for index, (
        build_seconds,
        forward_seconds,
        batch_size,
    ) in enumerate(
        zip(
            build_times,
            forward_times,
            batch_sizes,
        )
    ):
        core_seconds = float(
            build_seconds
            + forward_seconds
        )

        records.append(
            {
                "batch_index": index,
                "is_warmup": bool(
                    index < warmup_batches
                ),
                "batch_size": int(
                    batch_size
                ),
                "build_batch_tensors_seconds": (
                    float(
                        build_seconds
                    )
                ),
                "forward_allocator_seconds": (
                    float(
                        forward_seconds
                    )
                ),
                "core_pipeline_seconds": (
                    core_seconds
                ),
                "core_ms_per_spectrum": float(
                    1000.0
                    * core_seconds
                    / max(
                        1,
                        batch_size,
                    )
                ),
            }
        )

    per_batch = pd.DataFrame(
        records
    )

    measured = per_batch[
        ~per_batch["is_warmup"]
    ].copy()

    measured_spectra = int(
        measured["batch_size"].sum()
    )

    measured_core_seconds = float(
        measured[
            "core_pipeline_seconds"
        ].sum()
    )

    if measured_spectra <= 0:
        raise RuntimeError(
            "除去warmup后没有有效测试谱。"
        )

    if measured_core_seconds <= 0.0:
        raise RuntimeError(
            "有效计时时间为零。"
        )

    per_batch.to_csv(
        batch_path,
        index=False,
    )

    if (
        device.type == "cuda"
        and torch.cuda.is_available()
    ):
        synchronize(device)

        peak_allocated_bytes = int(
            torch.cuda
            .max_memory_allocated(
                device
            )
        )

        peak_reserved_bytes = int(
            torch.cuda
            .max_memory_reserved(
                device
            )
        )

        gpu_name = (
            torch.cuda
            .get_device_name(
                device
            )
        )

        gpu_total_memory_bytes = int(
            torch.cuda
            .get_device_properties(
                device
            )
            .total_memory
        )
    else:
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
        gpu_name = None
        gpu_total_memory_bytes = 0

    backbone_parameters = (
        parameter_statistics(
            backbone
        )
    )

    allocator_parameters = (
        parameter_statistics(
            allocator
        )
    )

    report = {
        "status": "complete",
        "benchmark_scope": (
            "locked final evaluation forward chain"
        ),
        "timing_definition": {
            "included": [
                "spectrum_allocator.build_batch_tensors",
                "spectrum_allocator.forward_allocator",
            ],
            "excluded_from_core_timing": [
                "validation pass",
                "dataloader iteration outside timed functions",
                "external CHUN Hungarian matching",
                "per-spectrum CSV serialization",
                "model and dataset initialization",
            ],
            "important_note": (
                "This is a conservative locked-evaluation "
                "forward-chain measurement. Any evaluation-side "
                "operations implemented inside the two timed "
                "functions remain included."
            ),
        },
        "dataset": {
            "test_spectrum_count": int(
                len(detail)
            ),
            "test_molecule_count": int(
                detail["spec_id"].nunique()
                if "mol_id" not in detail.columns
                else detail[
                    "mol_id"
                ].nunique()
            ),
            "batch_count": int(
                len(per_batch)
            ),
            "warmup_batch_count": int(
                warmup_batches
            ),
            "measured_batch_count": int(
                len(measured)
            ),
            "measured_spectrum_count": int(
                measured_spectra
            ),
            "median_batch_size": float(
                measured[
                    "batch_size"
                ].median()
            ),
        },
        "timing": {
            "core_pipeline_seconds": (
                measured_core_seconds
            ),
            "throughput_spectra_per_second": float(
                measured_spectra
                / measured_core_seconds
            ),
            "latency_ms_per_spectrum": float(
                1000.0
                * measured_core_seconds
                / measured_spectra
            ),
            "batch_latency_ms_mean": float(
                1000.0
                * measured[
                    "core_pipeline_seconds"
                ].mean()
            ),
            "batch_latency_ms_std": float(
                1000.0
                * measured[
                    "core_pipeline_seconds"
                ].std(
                    ddof=1
                )
            ),
            "batch_latency_ms_median": float(
                1000.0
                * measured[
                    "core_pipeline_seconds"
                ].median()
            ),
            "batch_latency_ms_p95": float(
                1000.0
                * measured[
                    "core_pipeline_seconds"
                ].quantile(
                    0.95
                )
            ),
            "full_test_evaluator_seconds": (
                test_evaluation_seconds
            ),
        },
        "gpu_memory": {
            "baseline_allocated_bytes": (
                baseline_allocated_bytes
            ),
            "baseline_allocated_gb": float(
                baseline_allocated_bytes
                / 1024**3
            ),
            "baseline_reserved_bytes": (
                baseline_reserved_bytes
            ),
            "baseline_reserved_gb": float(
                baseline_reserved_bytes
                / 1024**3
            ),
            "peak_allocated_bytes": (
                peak_allocated_bytes
            ),
            "peak_allocated_gb": float(
                peak_allocated_bytes
                / 1024**3
            ),
            "peak_reserved_bytes": (
                peak_reserved_bytes
            ),
            "peak_reserved_gb": float(
                peak_reserved_bytes
                / 1024**3
            ),
            "incremental_peak_allocated_bytes": int(
                max(
                    0,
                    peak_allocated_bytes
                    - baseline_allocated_bytes,
                )
            ),
            "incremental_peak_allocated_gb": float(
                max(
                    0,
                    peak_allocated_bytes
                    - baseline_allocated_bytes,
                )
                / 1024**3
            ),
        },
        "model": {
            "backbone_parameters": (
                backbone_parameters
            ),
            "allocator_parameters": (
                allocator_parameters
            ),
            "torch_parameter_total": int(
                backbone_parameters[
                    "total"
                ]
                + allocator_parameters[
                    "total"
                ]
            ),
            "regressor": (
                regressor_statistics(
                    regressor
                )
            ),
        },
        "hardware": {
            "gpu": gpu_name,
            "gpu_total_memory_bytes": (
                gpu_total_memory_bytes
            ),
            "gpu_total_memory_gb": float(
                gpu_total_memory_bytes
                / 1024**3
            ),
            "cpu": (
                platform.processor()
                or platform.machine()
            ),
            "platform": (
                platform.platform()
            ),
            "python": (
                platform.python_version()
            ),
            "torch": (
                torch.__version__
            ),
            "cuda_runtime": (
                torch.version.cuda
            ),
            "cuda_visible_devices": (
                os.environ.get(
                    "CUDA_VISIBLE_DEVICES"
                )
            ),
        },
        "artifacts": (
            artifact_inventory(
                seed_dir
            )
        ),
        "paths": {
            "seed_dir": str(
                seed_dir
            ),
            "output_dir": str(
                output_dir
            ),
            "per_batch_csv": str(
                batch_path
            ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark_state[
        "report_written"
    ] = True

    print()
    print("=" * 88)
    print("EXPERIMENT 7: EFFICIENCY BENCHMARK")
    print("=" * 88)
    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("WROTE:", report_path)
    print("WROTE:", batch_path)

    return metrics, detail


evaluator.evaluate_split = (
    timed_evaluate_split
)

sys.argv = [
    str(EVALUATOR_PATH),
    *evaluator_arguments,
]

evaluator.main()

if not benchmark_state.get(
    "report_written",
    False,
):
    raise RuntimeError(
        "没有捕获到测试集效率结果。"
    )

process_wall_seconds = float(
    time.perf_counter()
    - PROCESS_START_TIME
)

report = json.loads(
    report_path.read_text(
        encoding="utf-8"
    )
)

report[
    "full_process_wall_seconds"
] = process_wall_seconds

report[
    "full_process_wall_minutes"
] = (
    process_wall_seconds / 60.0
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "FULL PROCESS WALL TIME:",
    f"{process_wall_seconds:.3f} s",
)
print(
    "FINAL REPORT:",
    report_path,
)
