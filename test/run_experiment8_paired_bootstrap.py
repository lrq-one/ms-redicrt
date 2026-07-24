from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd().resolve()

BASELINE_FP = (
    ROOT
    / "runs/experiments/experiment8_bootstrap/inputs"
    / "baseline_per_spectrum_sanitized.csv.gz"
)

OUT_DIR = ROOT / "results_v2"
RUN_DIR = (
    ROOT
    / "runs/experiments/experiment8_bootstrap"
)

OURS_DIRS = {
    "random": (
        ROOT
        / "runs/experiments/molecule_disjoint_3seeds"
    ),
    "scaffold": (
        ROOT
        / "runs/experiments/scaffold_disjoint_3seeds"
    ),
}

EXPECTED = {
    "random": {
        "spectra": 3931,
        "molecules": 456,
    },
    "scaffold": {
        "spectra": 3960,
        "molecules": 450,
    },
}

BASELINES = [
    "neims",
    "massformer",
    "fragnnet_d3",
    "iceberg",
]

SEEDS = [42, 43, 44]
METRICS = ["cbin", "jss"]

BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 20260723
BOOTSTRAP_BATCH_SIZE = 1000
NUMERIC_TOLERANCE = 1e-6
ALPHA = 0.05

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(
            values
        ).tobytes()
    )
    return digest.hexdigest()


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    p_values = np.asarray(
        p_values,
        dtype=np.float64,
    )

    count = len(p_values)
    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    multipliers = (
        count
        - np.arange(
            count,
            dtype=np.float64,
        )
    )

    adjusted_sorted = (
        sorted_p
        * multipliers
    )

    adjusted_sorted = (
        np.maximum.accumulate(
            adjusted_sorted
        )
    )

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0.0,
        1.0,
    )

    adjusted = np.empty_like(
        adjusted_sorted
    )

    adjusted[
        order
    ] = adjusted_sorted

    return adjusted


def bootstrap_macro(
    differences: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        len(indices),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(indices),
        BOOTSTRAP_BATCH_SIZE,
    ):
        end = min(
            start + BOOTSTRAP_BATCH_SIZE,
            len(indices),
        )

        output[
            start:end
        ] = differences[
            indices[start:end]
        ].mean(
            axis=1
        )

    return output


def bootstrap_clustered_micro(
    molecule_score_sums: np.ndarray,
    molecule_spectrum_counts: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        len(indices),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(indices),
        BOOTSTRAP_BATCH_SIZE,
    ):
        end = min(
            start + BOOTSTRAP_BATCH_SIZE,
            len(indices),
        )

        batch_indices = indices[
            start:end
        ]

        numerators = (
            molecule_score_sums[
                batch_indices
            ].sum(
                axis=1
            )
        )

        denominators = (
            molecule_spectrum_counts[
                batch_indices
            ].sum(
                axis=1
            )
        )

        output[
            start:end
        ] = (
            numerators
            / denominators
        )

    return output


def summarize_bootstrap(
    distribution: np.ndarray,
    point_estimate: float,
) -> dict:
    ci_lower, ci_upper = np.quantile(
        distribution,
        [
            0.025,
            0.975,
        ],
    )

    probability_positive = float(
        np.mean(
            distribution > 0
        )
    )

    centered = (
        distribution
        - point_estimate
    )

    p_value = float(
        (
            np.sum(
                np.abs(
                    centered
                )
                >= abs(
                    point_estimate
                )
            )
            + 1
        )
        / (
            len(distribution)
            + 1
        )
    )

    return {
        "ci95_lower": float(
            ci_lower
        ),
        "ci95_upper": float(
            ci_upper
        ),
        "probability_delta_gt_zero": (
            probability_positive
        ),
        "p_bootstrap": p_value,
        "bootstrap_mean": float(
            distribution.mean()
        ),
        "bootstrap_std": float(
            distribution.std(
                ddof=1
            )
        ),
    }


def sanitize_metric(
    values: pd.Series,
    *,
    label: str,
) -> tuple[
    pd.Series,
    int,
    float,
]:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    if numeric.isna().any():
        raise RuntimeError(
            f"{label}存在NaN或非数值"
        )

    severe = (
        (numeric < -NUMERIC_TOLERANCE)
        | (
            numeric
            > 1.0
            + NUMERIC_TOLERANCE
        )
    )

    if severe.any():
        raise RuntimeError(
            f"{label}存在严重越界："
            f"min={numeric.min()}, "
            f"max={numeric.max()}"
        )

    clipped = numeric.clip(
        0.0,
        1.0,
    )

    correction = np.abs(
        clipped.to_numpy()
        - numeric.to_numpy()
    )

    return (
        clipped,
        int(
            np.sum(
                correction > 0
            )
        ),
        float(
            correction.max()
            if len(correction)
            else 0.0
        ),
    )


if not BASELINE_FP.is_file():
    raise FileNotFoundError(
        BASELINE_FP
    )


print("=" * 120)
print("EXPERIMENT 8: MOLECULE-LEVEL PAIRED BOOTSTRAP")
print("=" * 120)
print("BASELINE INPUT :", BASELINE_FP)
print(
    "BOOTSTRAP N   :",
    BOOTSTRAP_REPETITIONS,
)
print(
    "BOOTSTRAP SEED:",
    BOOTSTRAP_SEED,
)
print(
    "SEEDS AVERAGED:",
    SEEDS,
)


print()
print("=" * 120)
print("LOAD BASELINE INPUTS")
print("=" * 120)

baseline = pd.read_csv(
    BASELINE_FP
)

required_baseline_columns = {
    "split",
    "model",
    "seed",
    "spec_id",
    "mol_id",
    "ce",
    "cbin",
    "jss",
}

missing_baseline_columns = (
    required_baseline_columns
    - set(
        baseline.columns
    )
)

if missing_baseline_columns:
    raise RuntimeError(
        "基线输入缺少字段："
        f"{sorted(missing_baseline_columns)}"
    )

baseline["split"] = (
    baseline["split"]
    .astype(str)
    .str.lower()
)

baseline["model"] = (
    baseline["model"]
    .astype(str)
    .str.lower()
)

baseline["seed"] = pd.to_numeric(
    baseline["seed"],
    errors="raise",
).astype(int)

baseline["spec_id"] = pd.to_numeric(
    baseline["spec_id"],
    errors="raise",
).astype(int)

baseline["mol_id"] = pd.to_numeric(
    baseline["mol_id"],
    errors="raise",
).astype(int)

baseline["ce"] = pd.to_numeric(
    baseline["ce"],
    errors="raise",
).astype(float)

expected_baseline_groups = {
    (
        split,
        model,
        seed,
    )
    for split in EXPECTED
    for model in BASELINES
    for seed in SEEDS
}

actual_baseline_groups = set(
    zip(
        baseline["split"],
        baseline["model"],
        baseline["seed"],
    )
)

if (
    actual_baseline_groups
    != expected_baseline_groups
):
    raise RuntimeError(
        "基线24组不闭合。\n"
        f"Missing: "
        f"{sorted(expected_baseline_groups - actual_baseline_groups)}\n"
        f"Extra: "
        f"{sorted(actual_baseline_groups - expected_baseline_groups)}"
    )

print(
    "BASELINE ROWS :",
    len(baseline),
)

print(
    "BASELINE GROUPS:",
    len(actual_baseline_groups),
)


print()
print("=" * 120)
print("LOAD OURS TEST RESULTS")
print("=" * 120)

ours_frames = []
ours_input_manifest = []
ours_numeric_audit = []

for split, split_dir in OURS_DIRS.items():
    for seed in SEEDS:
        source_fp = (
            split_dir
            / f"seed_{seed}"
            / "v2e_full_063"
            / "final_locked_evaluation"
            / "test_per_spectrum_with_molecule.csv"
        )

        if not source_fp.is_file():
            raise FileNotFoundError(
                source_fp
            )

        frame = pd.read_csv(
            source_fp
        )

        required = {
            "spec_id",
            "mol_id",
            "ce",
            "cos",
            "jss",
        }

        missing = (
            required
            - set(
                frame.columns
            )
        )

        if missing:
            raise RuntimeError(
                f"{source_fp}缺少字段："
                f"{sorted(missing)}"
            )

        if (
            len(frame)
            != EXPECTED[
                split
            ]["spectra"]
        ):
            raise RuntimeError(
                f"{split}/Ours/seed{seed} "
                f"谱数错误：{len(frame)}"
            )

        frame["spec_id"] = (
            pd.to_numeric(
                frame["spec_id"],
                errors="raise",
            ).astype(int)
        )

        frame["mol_id"] = (
            pd.to_numeric(
                frame["mol_id"],
                errors="raise",
            ).astype(int)
        )

        frame["ce"] = (
            pd.to_numeric(
                frame["ce"],
                errors="raise",
            ).astype(float)
        )

        cbin, cbin_fix_count, cbin_max_fix = (
            sanitize_metric(
                frame["cos"],
                label=(
                    f"{split}/Ours/"
                    f"seed{seed}/CBIN"
                ),
            )
        )

        jss, jss_fix_count, jss_max_fix = (
            sanitize_metric(
                frame["jss"],
                label=(
                    f"{split}/Ours/"
                    f"seed{seed}/JSS"
                ),
            )
        )

        clean = pd.DataFrame({
            "split": split,
            "model": "ours",
            "seed": seed,
            "spec_id": frame[
                "spec_id"
            ],
            "mol_id": frame[
                "mol_id"
            ],
            "ce": frame["ce"],
            "cbin": cbin,
            "jss": jss,
        })

        if clean[
            "spec_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{split}/Ours/seed{seed}"
                "存在重复spec_id"
            )

        molecule_count = int(
            clean[
                "mol_id"
            ].nunique()
        )

        if (
            molecule_count
            != EXPECTED[
                split
            ]["molecules"]
        ):
            raise RuntimeError(
                f"{split}/Ours/seed{seed} "
                f"分子数错误：{molecule_count}"
            )

        ours_frames.append(
            clean
        )

        ours_input_manifest.append({
            "split": split,
            "model": "ours",
            "seed": seed,
            "source_file": str(
                source_fp
            ),
            "sha256": sha256_file(
                source_fp
            ),
            "rows": len(clean),
            "molecules": molecule_count,
        })

        ours_numeric_audit.append({
            "split": split,
            "seed": seed,
            "cbin_correction_count": (
                cbin_fix_count
            ),
            "cbin_max_correction": (
                cbin_max_fix
            ),
            "jss_correction_count": (
                jss_fix_count
            ),
            "jss_max_correction": (
                jss_max_fix
            ),
        })

        print(
            f"{split:9s} "
            f"ours seed={seed} "
            f"rows={len(clean)} "
            f"mols={molecule_count} "
            f"cbin_fix={cbin_fix_count} "
            f"jss_fix={jss_fix_count}"
        )

ours = pd.concat(
    ours_frames,
    ignore_index=True,
)


print()
print("=" * 120)
print("CROSS-MODEL ALIGNMENT AUDIT")
print("=" * 120)

all_seed_level = pd.concat(
    [
        ours,
        baseline,
    ],
    ignore_index=True,
)

all_seed_level = all_seed_level[
    [
        "split",
        "model",
        "seed",
        "spec_id",
        "mol_id",
        "ce",
        "cbin",
        "jss",
    ]
].copy()

for split in EXPECTED:
    reference = (
        ours[
            (
                ours["split"]
                == split
            )
            & (
                ours["seed"]
                == SEEDS[0]
            )
        ][
            [
                "spec_id",
                "mol_id",
                "ce",
            ]
        ]
        .sort_values(
            "spec_id"
        )
        .reset_index(
            drop=True
        )
    )

    for model in [
        "ours",
        *BASELINES,
    ]:
        for seed in SEEDS:
            current = (
                all_seed_level[
                    (
                        all_seed_level[
                            "split"
                        ]
                        == split
                    )
                    & (
                        all_seed_level[
                            "model"
                        ]
                        == model
                    )
                    & (
                        all_seed_level[
                            "seed"
                        ]
                        == seed
                    )
                ][
                    [
                        "spec_id",
                        "mol_id",
                        "ce",
                    ]
                ]
                .sort_values(
                    "spec_id"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(current) != len(
                reference
            ):
                raise RuntimeError(
                    f"{split}/{model}/"
                    f"seed{seed}行数不一致"
                )

            if not np.array_equal(
                current[
                    "spec_id"
                ].to_numpy(),
                reference[
                    "spec_id"
                ].to_numpy(),
            ):
                raise RuntimeError(
                    f"{split}/{model}/"
                    f"seed{seed}的"
                    "spec_id集合不一致"
                )

            if not np.array_equal(
                current[
                    "mol_id"
                ].to_numpy(),
                reference[
                    "mol_id"
                ].to_numpy(),
            ):
                raise RuntimeError(
                    f"{split}/{model}/"
                    f"seed{seed}的"
                    "spec_id→mol_id映射不一致"
                )

            if not np.allclose(
                current[
                    "ce"
                ].to_numpy(),
                reference[
                    "ce"
                ].to_numpy(),
                rtol=0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"{split}/{model}/"
                    f"seed{seed}的CE映射不一致"
                )

    print(
        f"{split:9s}: "
        f"{EXPECTED[split]['spectra']} spectra, "
        f"{EXPECTED[split]['molecules']} molecules, "
        "5 models × 3 seeds aligned"
    )


print()
print("=" * 120)
print("AVERAGE THREE SEEDS BEFORE BOOTSTRAP")
print("=" * 120)

seed_averaged = (
    all_seed_level.groupby(
        [
            "split",
            "model",
            "spec_id",
            "mol_id",
        ],
        as_index=False,
        sort=True,
    )
    .agg(
        ce=(
            "ce",
            "first",
        ),
        cbin=(
            "cbin",
            "mean",
        ),
        jss=(
            "jss",
            "mean",
        ),
        seed_count=(
            "seed",
            "nunique",
        ),
    )
)

if not (
    seed_averaged[
        "seed_count"
    ]
    == 3
).all():
    raise RuntimeError(
        "存在未覆盖3个种子的谱图"
    )

seed_averaged.drop(
    columns=[
        "seed_count",
    ],
    inplace=True,
)

per_molecule = (
    seed_averaged.groupby(
        [
            "split",
            "model",
            "mol_id",
        ],
        as_index=False,
        sort=True,
    )
    .agg(
        spec_count=(
            "spec_id",
            "size",
        ),
        mean_ce=(
            "ce",
            "mean",
        ),
        cbin=(
            "cbin",
            "mean",
        ),
        jss=(
            "jss",
            "mean",
        ),
    )
)

for split in EXPECTED:
    for model in [
        "ours",
        *BASELINES,
    ]:
        spectrum_count = int(
            (
                seed_averaged[
                    (
                        seed_averaged[
                            "split"
                        ]
                        == split
                    )
                    & (
                        seed_averaged[
                            "model"
                        ]
                        == model
                    )
                ]
            ).shape[0]
        )

        molecule_count = int(
            (
                per_molecule[
                    (
                        per_molecule[
                            "split"
                        ]
                        == split
                    )
                    & (
                        per_molecule[
                            "model"
                        ]
                        == model
                    )
                ]
            ).shape[0]
        )

        if (
            spectrum_count
            != EXPECTED[
                split
            ]["spectra"]
        ):
            raise RuntimeError(
                f"{split}/{model}: "
                "种子平均后谱数错误"
            )

        if (
            molecule_count
            != EXPECTED[
                split
            ]["molecules"]
        ):
            raise RuntimeError(
                f"{split}/{model}: "
                "种子平均后分子数错误"
            )

print(
    "SEED-AVERAGED SPECTRUM ROWS:",
    len(seed_averaged),
)

print(
    "PER-MOLECULE ROWS:",
    len(per_molecule),
)


summary_rows = []
pairwise_molecule_frames = []
bootstrap_distributions = {}
bootstrap_index_audit = {}

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


print()
print("=" * 120)
print("RUN 20,000 MOLECULE-LEVEL PAIRED BOOTSTRAPS")
print("=" * 120)

for split in [
    "random",
    "scaffold",
]:
    ours_molecule = (
        per_molecule[
            (
                per_molecule[
                    "split"
                ]
                == split
            )
            & (
                per_molecule[
                    "model"
                ]
                == "ours"
            )
        ]
        .sort_values(
            "mol_id"
        )
        .reset_index(
            drop=True
        )
    )

    molecule_ids = (
        ours_molecule[
            "mol_id"
        ].to_numpy(
            dtype=np.int64
        )
    )

    molecule_count = len(
        molecule_ids
    )

    bootstrap_indices = rng.integers(
        low=0,
        high=molecule_count,
        size=(
            BOOTSTRAP_REPETITIONS,
            molecule_count,
        ),
        dtype=np.int32,
    )

    bootstrap_index_audit[
        split
    ] = {
        "shape": list(
            bootstrap_indices.shape
        ),
        "sha256": sha256_array(
            bootstrap_indices
        ),
    }

    ours_spectrum = (
        seed_averaged[
            (
                seed_averaged[
                    "split"
                ]
                == split
            )
            & (
                seed_averaged[
                    "model"
                ]
                == "ours"
            )
        ]
        .sort_values(
            "spec_id"
        )
        .reset_index(
            drop=True
        )
    )

    for baseline_name in BASELINES:
        baseline_molecule = (
            per_molecule[
                (
                    per_molecule[
                        "split"
                    ]
                    == split
                )
                & (
                    per_molecule[
                        "model"
                    ]
                    == baseline_name
                )
            ]
            .sort_values(
                "mol_id"
            )
            .reset_index(
                drop=True
            )
        )

        if not np.array_equal(
            baseline_molecule[
                "mol_id"
            ].to_numpy(),
            molecule_ids,
        ):
            raise RuntimeError(
                f"{split}/{baseline_name}: "
                "分子集合不一致"
            )

        if not np.array_equal(
            baseline_molecule[
                "spec_count"
            ].to_numpy(),
            ours_molecule[
                "spec_count"
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"{split}/{baseline_name}: "
                "每分子谱数不一致"
            )

        baseline_spectrum = (
            seed_averaged[
                (
                    seed_averaged[
                        "split"
                    ]
                    == split
                )
                & (
                    seed_averaged[
                        "model"
                    ]
                    == baseline_name
                )
            ]
            .sort_values(
                "spec_id"
            )
            .reset_index(
                drop=True
            )
        )

        if not np.array_equal(
            baseline_spectrum[
                "spec_id"
            ].to_numpy(),
            ours_spectrum[
                "spec_id"
            ].to_numpy(),
        ):
            raise RuntimeError(
                f"{split}/{baseline_name}: "
                "逐谱集合不一致"
            )

        for metric in METRICS:
            ours_molecule_scores = (
                ours_molecule[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
            )

            baseline_molecule_scores = (
                baseline_molecule[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
            )

            molecule_differences = (
                ours_molecule_scores
                - baseline_molecule_scores
            )

            pairwise_molecule_frames.append(
                pd.DataFrame({
                    "split": split,
                    "baseline": (
                        baseline_name
                    ),
                    "metric": metric,
                    "mol_id": (
                        molecule_ids
                    ),
                    "spec_count": (
                        ours_molecule[
                            "spec_count"
                        ].to_numpy()
                    ),
                    "ours_score": (
                        ours_molecule_scores
                    ),
                    "baseline_score": (
                        baseline_molecule_scores
                    ),
                    "paired_delta": (
                        molecule_differences
                    ),
                })
            )

            macro_point = float(
                molecule_differences.mean()
            )

            macro_distribution = (
                bootstrap_macro(
                    molecule_differences,
                    bootstrap_indices,
                )
            )

            macro_stats = (
                summarize_bootstrap(
                    macro_distribution,
                    macro_point,
                )
            )

            macro_key = (
                f"{split}__"
                f"{baseline_name}__"
                f"{metric}__"
                "molecule_macro"
            )

            bootstrap_distributions[
                macro_key
            ] = macro_distribution

            ours_macro_mean = float(
                ours_molecule_scores.mean()
            )

            baseline_macro_mean = float(
                baseline_molecule_scores.mean()
            )

            summary_rows.append({
                "split": split,
                "baseline": baseline_name,
                "metric": metric,
                "aggregation": (
                    "molecule_macro"
                ),
                "n_spectra": EXPECTED[
                    split
                ]["spectra"],
                "n_molecules": (
                    molecule_count
                ),
                "ours_mean": (
                    ours_macro_mean
                ),
                "baseline_mean": (
                    baseline_macro_mean
                ),
                "absolute_delta": (
                    macro_point
                ),
                "relative_gain_percent": (
                    100.0
                    * macro_point
                    / baseline_macro_mean
                    if baseline_macro_mean
                    != 0
                    else np.nan
                ),
                "molecule_win_rate": float(
                    np.mean(
                        molecule_differences
                        > 0
                    )
                ),
                "molecule_tie_rate": float(
                    np.mean(
                        np.isclose(
                            molecule_differences,
                            0.0,
                            atol=1e-12,
                            rtol=0,
                        )
                    )
                ),
                "median_paired_delta": float(
                    np.median(
                        molecule_differences
                    )
                ),
                **macro_stats,
            })

            spectrum_differences = (
                ours_spectrum[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
                - baseline_spectrum[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
            )

            spectrum_difference_df = (
                pd.DataFrame({
                    "mol_id": (
                        ours_spectrum[
                            "mol_id"
                        ].to_numpy(
                            dtype=np.int64
                        )
                    ),
                    "difference": (
                        spectrum_differences
                    ),
                })
            )

            clustered = (
                spectrum_difference_df.groupby(
                    "mol_id",
                    sort=True,
                )[
                    "difference"
                ]
                .agg(
                    [
                        "sum",
                        "size",
                    ]
                )
                .reset_index()
            )

            if not np.array_equal(
                clustered[
                    "mol_id"
                ].to_numpy(
                    dtype=np.int64
                ),
                molecule_ids,
            ):
                raise RuntimeError(
                    f"{split}/{baseline_name}/"
                    f"{metric}: "
                    "clustered micro分子顺序错误"
                )

            molecule_score_sums = (
                clustered[
                    "sum"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            molecule_spectrum_counts = (
                clustered[
                    "size"
                ].to_numpy(
                    dtype=np.float64
                )
            )

            micro_point = float(
                spectrum_differences.mean()
            )

            micro_distribution = (
                bootstrap_clustered_micro(
                    molecule_score_sums,
                    molecule_spectrum_counts,
                    bootstrap_indices,
                )
            )

            micro_stats = (
                summarize_bootstrap(
                    micro_distribution,
                    micro_point,
                )
            )

            micro_key = (
                f"{split}__"
                f"{baseline_name}__"
                f"{metric}__"
                "clustered_micro"
            )

            bootstrap_distributions[
                micro_key
            ] = micro_distribution

            ours_micro_mean = float(
                ours_spectrum[
                    metric
                ].mean()
            )

            baseline_micro_mean = float(
                baseline_spectrum[
                    metric
                ].mean()
            )

            summary_rows.append({
                "split": split,
                "baseline": baseline_name,
                "metric": metric,
                "aggregation": (
                    "clustered_micro"
                ),
                "n_spectra": EXPECTED[
                    split
                ]["spectra"],
                "n_molecules": (
                    molecule_count
                ),
                "ours_mean": (
                    ours_micro_mean
                ),
                "baseline_mean": (
                    baseline_micro_mean
                ),
                "absolute_delta": (
                    micro_point
                ),
                "relative_gain_percent": (
                    100.0
                    * micro_point
                    / baseline_micro_mean
                    if baseline_micro_mean
                    != 0
                    else np.nan
                ),
                "molecule_win_rate": float(
                    np.mean(
                        molecule_differences
                        > 0
                    )
                ),
                "molecule_tie_rate": float(
                    np.mean(
                        np.isclose(
                            molecule_differences,
                            0.0,
                            atol=1e-12,
                            rtol=0,
                        )
                    )
                ),
                "median_paired_delta": float(
                    np.median(
                        molecule_differences
                    )
                ),
                **micro_stats,
            })

            print(
                f"{split:9s} "
                f"{baseline_name:12s} "
                f"{metric:4s} "
                f"macro Δ={macro_point:+.6f} "
                f"[{macro_stats['ci95_lower']:+.6f}, "
                f"{macro_stats['ci95_upper']:+.6f}] "
                f"micro Δ={micro_point:+.6f} "
                f"[{micro_stats['ci95_lower']:+.6f}, "
                f"{micro_stats['ci95_upper']:+.6f}]"
            )


summary = pd.DataFrame(
    summary_rows
)

summary["p_holm"] = np.nan

for (
    metric,
    aggregation,
), indices in summary.groupby(
    [
        "metric",
        "aggregation",
    ]
).groups.items():
    group_indices = list(
        indices
    )

    if len(group_indices) != 8:
        raise RuntimeError(
            f"{metric}/{aggregation}: "
            f"Holm检验族不是8项，"
            f"实际={len(group_indices)}"
        )

    adjusted = holm_adjust(
        summary.loc[
            group_indices,
            "p_bootstrap",
        ].to_numpy(
            dtype=np.float64
        )
    )

    summary.loc[
        group_indices,
        "p_holm",
    ] = adjusted

summary[
    "ci_excludes_zero"
] = (
    (
        summary[
            "ci95_lower"
        ] > 0
    )
    | (
        summary[
            "ci95_upper"
        ] < 0
    )
)

summary[
    "significant_after_holm"
] = (
    summary[
        "p_holm"
    ] < ALPHA
)

summary[
    "direction"
] = np.where(
    summary[
        "absolute_delta"
    ] > 0,
    "ours_better",
    np.where(
        summary[
            "absolute_delta"
        ] < 0,
        "baseline_better",
        "tie",
    ),
)

summary[
    "bootstrap_repetitions"
] = BOOTSTRAP_REPETITIONS

summary[
    "bootstrap_seed"
] = BOOTSTRAP_SEED

summary = summary.sort_values(
    [
        "aggregation",
        "metric",
        "split",
        "baseline",
    ]
).reset_index(
    drop=True
)

macro_summary = (
    summary[
        summary[
            "aggregation"
        ]
        == "molecule_macro"
    ]
    .reset_index(
        drop=True
    )
)

micro_summary = (
    summary[
        summary[
            "aggregation"
        ]
        == "clustered_micro"
    ]
    .reset_index(
        drop=True
    )
)

pairwise_molecule = pd.concat(
    pairwise_molecule_frames,
    ignore_index=True,
)

seed_averaged_fp = (
    OUT_DIR
    / "experiment8_per_spectrum_seed_averaged.csv.gz"
)

per_molecule_fp = (
    OUT_DIR
    / "experiment8_per_molecule_scores.csv.gz"
)

pairwise_molecule_fp = (
    OUT_DIR
    / "experiment8_pairwise_molecule_differences.csv.gz"
)

macro_fp = (
    OUT_DIR
    / "experiment8_bootstrap_macro.csv"
)

micro_fp = (
    OUT_DIR
    / "experiment8_bootstrap_clustered_micro.csv"
)

all_summary_fp = (
    OUT_DIR
    / "experiment8_bootstrap_all.csv"
)

distribution_fp = (
    OUT_DIR
    / "experiment8_bootstrap_distributions.npz"
)

audit_fp = (
    OUT_DIR
    / "experiment8_bootstrap_audit.json"
)

seed_averaged.to_csv(
    seed_averaged_fp,
    index=False,
    compression="gzip",
)

per_molecule.to_csv(
    per_molecule_fp,
    index=False,
    compression="gzip",
)

pairwise_molecule.to_csv(
    pairwise_molecule_fp,
    index=False,
    compression="gzip",
)

macro_summary.to_csv(
    macro_fp,
    index=False,
)

micro_summary.to_csv(
    micro_fp,
    index=False,
)

summary.to_csv(
    all_summary_fp,
    index=False,
)

np.savez_compressed(
    distribution_fp,
    **bootstrap_distributions,
)


audit = {
    "created_at_utc": utc_now(),
    "status": "OK",
    "analysis": (
        "molecule-level paired bootstrap "
        "with seeds averaged before resampling"
    ),
    "baseline_input": str(
        BASELINE_FP
    ),
    "baseline_input_sha256": (
        sha256_file(
            BASELINE_FP
        )
    ),
    "ours_inputs": (
        ours_input_manifest
    ),
    "ours_numeric_audit": (
        ours_numeric_audit
    ),
    "splits": EXPECTED,
    "models": {
        "ours": "ours",
        "baselines": BASELINES,
    },
    "seeds": SEEDS,
    "metrics": METRICS,
    "bootstrap": {
        "sampling_unit": "molecule",
        "repetitions": (
            BOOTSTRAP_REPETITIONS
        ),
        "random_seed": (
            BOOTSTRAP_SEED
        ),
        "confidence_interval": (
            "percentile 95%"
        ),
        "macro_definition": (
            "mean paired difference across "
            "seed-averaged molecule scores"
        ),
        "clustered_micro_definition": (
            "resample molecules with replacement, "
            "carry all spectra of each sampled "
            "molecule, and recompute the "
            "spectrum-weighted paired mean"
        ),
        "p_value_definition": (
            "two-sided centered-bootstrap test "
            "with finite-sample correction"
        ),
        "holm_family": (
            "8 split-by-baseline comparisons "
            "within each metric-by-aggregation "
            "family"
        ),
        "index_audit": (
            bootstrap_index_audit
        ),
    },
    "row_counts": {
        "seed_level_total": int(
            len(
                all_seed_level
            )
        ),
        "seed_averaged_spectrum_total": int(
            len(
                seed_averaged
            )
        ),
        "per_molecule_total": int(
            len(
                per_molecule
            )
        ),
        "summary_rows": int(
            len(
                summary
            )
        ),
        "macro_rows": int(
            len(
                macro_summary
            )
        ),
        "clustered_micro_rows": int(
            len(
                micro_summary
            )
        ),
    },
    "outputs": {
        "macro": str(
            macro_fp
        ),
        "clustered_micro": str(
            micro_fp
        ),
        "all_summary": str(
            all_summary_fp
        ),
        "seed_averaged_per_spectrum": str(
            seed_averaged_fp
        ),
        "per_molecule_scores": str(
            per_molecule_fp
        ),
        "pairwise_molecule_differences": str(
            pairwise_molecule_fp
        ),
        "bootstrap_distributions": str(
            distribution_fp
        ),
    },
}

audit_fp.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print()
print("=" * 120)
print("EXPERIMENT 8 MOLECULE-MACRO RESULTS")
print("=" * 120)

print(
    macro_summary[
        [
            "split",
            "metric",
            "baseline",
            "ours_mean",
            "baseline_mean",
            "absolute_delta",
            "ci95_lower",
            "ci95_upper",
            "p_bootstrap",
            "p_holm",
            "molecule_win_rate",
            "significant_after_holm",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 120)
print("EXPERIMENT 8 CLUSTERED-MICRO RESULTS")
print("=" * 120)

print(
    micro_summary[
        [
            "split",
            "metric",
            "baseline",
            "ours_mean",
            "baseline_mean",
            "absolute_delta",
            "ci95_lower",
            "ci95_upper",
            "p_bootstrap",
            "p_holm",
            "significant_after_holm",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 120)
print("EXPERIMENT 8 OUTPUTS")
print("=" * 120)
print("MACRO SUMMARY :", macro_fp)
print("MICRO SUMMARY :", micro_fp)
print("ALL RESULTS   :", all_summary_fp)
print("DISTRIBUTIONS :", distribution_fp)
print("AUDIT         :", audit_fp)
print()

if (
    len(macro_summary) == 16
    and len(micro_summary) == 16
    and len(summary) == 32
):
    print(
        "EXPERIMENT8_MOLECULE_PAIRED_BOOTSTRAP_COMPLETE"
    )
else:
    raise RuntimeError(
        "实验8结果行数不闭合"
    )
