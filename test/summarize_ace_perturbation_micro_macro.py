from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd()

EXPERIMENT_ROOT = (
    ROOT
    / "runs"
    / "experiments"
    / "molecule_disjoint_3seeds"
)

PROC_PATH = (
    ROOT
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
    / "spec_df.pkl"
)

SPLIT_ROOT = (
    ROOT
    / "data"
    / "split"
    / "nist20_qtof_cid_safe19659_qcv1_trainonly"
)

OUTPUT_ROOT = (
    EXPERIMENT_ROOT
    / "ace_perturbation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = (42, 43, 44)

MODES = (
    "true",
    "median",
    "shuffled",
)

METRICS = {
    "cos_0.01": "cbin_0.01",
    "jss_0.01": "jss_0.01",
    "chun_10ppm": "chun_10ppm",
}


def load_table(path: Path) -> Any:
    name = path.name.lower()

    if (
        name.endswith(".pkl")
        or name.endswith(".pkl.gz")
        or name.endswith(".pickle")
        or name.endswith(".pickle.gz")
    ):
        return pd.read_pickle(path)

    if (
        name.endswith(".csv")
        or name.endswith(".csv.gz")
    ):
        return pd.read_csv(path)

    if name.endswith(".npy"):
        return np.load(
            path,
            allow_pickle=True,
        )

    if name.endswith(".json"):
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    raise ValueError(
        f"Unsupported table: {path}"
    )


def to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        return obj.to_frame()

    if isinstance(obj, dict):
        try:
            return pd.DataFrame(obj)
        except Exception:
            for key, value in obj.items():
                if "train" in str(key).lower():
                    return to_dataframe(value)

    if isinstance(
        obj,
        (list, tuple, np.ndarray),
    ):
        array = np.asarray(
            obj,
            dtype=object,
        ).reshape(-1)

        return pd.DataFrame(
            {"id": array}
        )

    raise TypeError(
        f"Cannot convert to DataFrame: {type(obj)}"
    )


def numeric_ids(series: pd.Series) -> set[int]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return set(
        values.astype(int).tolist()
    )


def find_detail_file(
    seed: int,
    mode: str,
) -> Path:
    seed_root = (
        EXPERIMENT_ROOT
        / f"seed_{seed}"
    )

    if mode == "true":
        base_dir = (
            seed_root
            / "chun_10ppm"
        )
    else:
        base_dir = (
            seed_root
            / "ace_perturbation"
            / mode
        )

    preferred = (
        base_dir
        / "test_per_spectrum_metrics.csv"
    )

    if preferred.is_file():
        return preferred

    required = {
        "spec_id",
        "ce",
        *METRICS.keys(),
    }

    candidates = []

    if base_dir.is_dir():
        for path in base_dir.rglob("*.csv"):
            try:
                columns = set(
                    pd.read_csv(
                        path,
                        nrows=0,
                    ).columns
                )
            except Exception:
                continue

            if required.issubset(columns):
                candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"seed={seed}, mode={mode}: "
            f"找不到逐谱测试结果，目录={base_dir}"
        )

    raise RuntimeError(
        f"seed={seed}, mode={mode}: "
        f"找到多个候选逐谱文件：{candidates}"
    )


def resolve_training_spectra(
    spec_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame | None,
    dict[str, Any],
]:
    if not SPLIT_ROOT.is_dir():
        return None, {
            "status": "split_directory_missing",
            "split_root": str(SPLIT_ROOT),
        }

    supported_suffixes = (
        ".csv",
        ".csv.gz",
        ".pkl",
        ".pkl.gz",
        ".pickle",
        ".pickle.gz",
        ".npy",
        ".json",
    )

    candidates = [
        path
        for path in SPLIT_ROOT.rglob("*")
        if (
            path.is_file()
            and "train" in path.name.lower()
            and any(
                path.name.lower().endswith(
                    suffix
                )
                for suffix
                in supported_suffixes
            )
        )
    ]

    spec_id_set = set(
        pd.to_numeric(
            spec_df["spec_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    mol_id_set = set(
        pd.to_numeric(
            spec_df["mol_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    attempts = []

    for path in sorted(candidates):
        try:
            table = to_dataframe(
                load_table(path)
            )
        except Exception as error:
            attempts.append(
                {
                    "path": str(path),
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )
            continue

        table = table.drop(
            columns=[
                column
                for column in table.columns
                if str(column).lower().startswith(
                    "unnamed:"
                )
            ],
            errors="ignore",
        )

        column_map = {
            str(column).lower(): column
            for column in table.columns
        }

        if "spec_id" in column_map:
            ids = numeric_ids(
                table[
                    column_map["spec_id"]
                ]
            )

            overlap = ids & spec_id_set

            if overlap:
                subset = spec_df[
                    spec_df["spec_id"].isin(
                        overlap
                    )
                ].copy()

                return subset, {
                    "status": "resolved",
                    "path": str(path),
                    "id_type": "spec_id",
                    "id_count": len(ids),
                    "matched_spectra": len(
                        subset
                    ),
                }

        if "mol_id" in column_map:
            ids = numeric_ids(
                table[
                    column_map["mol_id"]
                ]
            )

            overlap = ids & mol_id_set

            if overlap:
                subset = spec_df[
                    spec_df["mol_id"].isin(
                        overlap
                    )
                ].copy()

                return subset, {
                    "status": "resolved",
                    "path": str(path),
                    "id_type": "mol_id",
                    "id_count": len(ids),
                    "matched_spectra": len(
                        subset
                    ),
                }

        for column in table.columns:
            ids = numeric_ids(
                table[column]
            )

            if not ids:
                continue

            spec_overlap = (
                ids & spec_id_set
            )

            mol_overlap = (
                ids & mol_id_set
            )

            spec_fraction = (
                len(spec_overlap)
                / max(1, len(ids))
            )

            mol_fraction = (
                len(mol_overlap)
                / max(1, len(ids))
            )

            if (
                spec_fraction >= 0.90
                and spec_fraction
                >= mol_fraction
            ):
                subset = spec_df[
                    spec_df["spec_id"].isin(
                        spec_overlap
                    )
                ].copy()

                return subset, {
                    "status": "resolved",
                    "path": str(path),
                    "id_type": (
                        f"inferred_spec_id:"
                        f"{column}"
                    ),
                    "id_count": len(ids),
                    "matched_spectra": len(
                        subset
                    ),
                }

            if mol_fraction >= 0.90:
                subset = spec_df[
                    spec_df["mol_id"].isin(
                        mol_overlap
                    )
                ].copy()

                return subset, {
                    "status": "resolved",
                    "path": str(path),
                    "id_type": (
                        f"inferred_mol_id:"
                        f"{column}"
                    ),
                    "id_count": len(ids),
                    "matched_spectra": len(
                        subset
                    ),
                }

        attempts.append(
            {
                "path": str(path),
                "columns": [
                    str(column)
                    for column
                    in table.columns
                ],
                "rows": int(len(table)),
            }
        )

    return None, {
        "status": "unresolved",
        "split_root": str(SPLIT_ROOT),
        "candidate_files": [
            str(path)
            for path in candidates
        ],
        "attempts": attempts,
    }


if not PROC_PATH.is_file():
    raise FileNotFoundError(
        PROC_PATH
    )

spec_df = pd.read_pickle(
    PROC_PATH
)

required_spec_columns = {
    "spec_id",
    "mol_id",
    "ace",
}

missing_spec_columns = (
    required_spec_columns
    - set(spec_df.columns)
)

if missing_spec_columns:
    raise RuntimeError(
        "spec_df缺少字段："
        f"{sorted(missing_spec_columns)}"
    )

mapping = (
    spec_df[
        ["spec_id", "mol_id"]
    ]
    .drop_duplicates(
        subset=["spec_id"]
    )
    .copy()
)

train_spec_df, train_resolution = (
    resolve_training_spectra(
        spec_df
    )
)

if train_spec_df is not None:
    training_ace = pd.to_numeric(
        train_spec_df["ace"],
        errors="coerce",
    ).dropna()

    if training_ace.empty:
        training_median = None
    else:
        training_median = float(
            training_ace.median()
        )
else:
    training_median = None

print()
print("=" * 88)
print("TRAINING ACE MEDIAN CHECK")
print("=" * 88)
print(json.dumps(
    {
        "training_split_resolution": (
            train_resolution
        ),
        "training_ace_median": (
            training_median
        ),
    },
    indent=2,
    ensure_ascii=False,
))

seed_rows = []
details: dict[
    tuple[int, str],
    pd.DataFrame,
] = {}

audit_rows = []

for seed in SEEDS:
    for mode in MODES:
        detail_path = find_detail_file(
            seed=seed,
            mode=mode,
        )

        detail = pd.read_csv(
            detail_path
        )

        required_columns = {
            "spec_id",
            "ce",
            *METRICS.keys(),
        }

        missing = (
            required_columns
            - set(detail.columns)
        )

        if missing:
            raise RuntimeError(
                f"{detail_path}缺少字段："
                f"{sorted(missing)}"
            )

        if len(detail) != 3931:
            raise RuntimeError(
                f"seed={seed}, mode={mode}: "
                f"谱数={len(detail)}，应为3931"
            )

        if (
            detail["spec_id"].nunique()
            != 3931
        ):
            raise RuntimeError(
                f"seed={seed}, mode={mode}: "
                "spec_id不唯一"
            )

        merged = detail.merge(
            mapping,
            on="spec_id",
            how="left",
            validate="one_to_one",
        )

        if merged["mol_id"].isna().any():
            missing_count = int(
                merged["mol_id"]
                .isna()
                .sum()
            )

            raise RuntimeError(
                f"seed={seed}, mode={mode}: "
                f"{missing_count}条谱无法映射mol_id"
            )

        per_molecule = (
            merged
            .groupby(
                "mol_id",
                as_index=False,
            )[list(METRICS.keys())]
            .mean()
        )

        if len(per_molecule) != 456:
            raise RuntimeError(
                f"seed={seed}, mode={mode}: "
                f"分子数={len(per_molecule)}，"
                "应为456"
            )

        mode_output_dir = (
            EXPERIMENT_ROOT
            / f"seed_{seed}"
            / "ace_perturbation"
            / mode
        )

        mode_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        per_molecule_path = (
            mode_output_dir
            / "test_per_molecule_metrics.csv"
        )

        per_molecule.to_csv(
            per_molecule_path,
            index=False,
        )

        row = {
            "seed": seed,
            "mode": mode,
            "spectrum_count": int(
                len(merged)
            ),
            "molecule_count": int(
                len(per_molecule)
            ),
            "ace_min": float(
                pd.to_numeric(
                    merged["ce"],
                    errors="coerce",
                ).min()
            ),
            "ace_max": float(
                pd.to_numeric(
                    merged["ce"],
                    errors="coerce",
                ).max()
            ),
            "ace_unique_count": int(
                pd.to_numeric(
                    merged["ce"],
                    errors="coerce",
                ).nunique()
            ),
            "detail_path": str(
                detail_path
            ),
            "per_molecule_path": str(
                per_molecule_path
            ),
        }

        for source, label in METRICS.items():
            row[
                f"micro_{label}"
            ] = float(
                merged[source].mean()
            )

            row[
                f"macro_{label}"
            ] = float(
                per_molecule[
                    source
                ].mean()
            )

        seed_rows.append(row)
        details[(seed, mode)] = (
            merged.copy()
        )

for seed in SEEDS:
    true_detail = (
        details[(seed, "true")]
        .sort_values("spec_id")
        .reset_index(drop=True)
    )

    median_detail = (
        details[(seed, "median")]
        .sort_values("spec_id")
        .reset_index(drop=True)
    )

    shuffled_detail = (
        details[(seed, "shuffled")]
        .sort_values("spec_id")
        .reset_index(drop=True)
    )

    same_true_median_ids = bool(
        np.array_equal(
            true_detail["spec_id"].to_numpy(),
            median_detail["spec_id"].to_numpy(),
        )
    )

    same_true_shuffled_ids = bool(
        np.array_equal(
            true_detail["spec_id"].to_numpy(),
            shuffled_detail[
                "spec_id"
            ].to_numpy(),
        )
    )

    true_ce = pd.to_numeric(
        true_detail["ce"],
        errors="coerce",
    ).to_numpy(dtype=float)

    median_ce = pd.to_numeric(
        median_detail["ce"],
        errors="coerce",
    ).to_numpy(dtype=float)

    shuffled_ce = pd.to_numeric(
        shuffled_detail["ce"],
        errors="coerce",
    ).to_numpy(dtype=float)

    median_unique = np.unique(
        median_ce[
            np.isfinite(median_ce)
        ]
    )

    median_is_constant = bool(
        len(median_unique) == 1
    )

    median_constant = (
        float(median_unique[0])
        if median_is_constant
        else None
    )

    shuffled_multiset_preserved = bool(
        np.allclose(
            np.sort(true_ce),
            np.sort(shuffled_ce),
            atol=1.0e-12,
            rtol=0.0,
        )
    )

    shuffled_mapping_changed = bool(
        not np.allclose(
            true_ce,
            shuffled_ce,
            atol=1.0e-12,
            rtol=0.0,
        )
    )

    median_matches_training = (
        bool(
            training_median is not None
            and median_constant is not None
            and abs(
                training_median
                - median_constant
            )
            <= 1.0e-12
        )
    )

    audit_rows.append(
        {
            "seed": seed,
            "same_spec_ids_true_median": (
                same_true_median_ids
            ),
            "same_spec_ids_true_shuffled": (
                same_true_shuffled_ids
            ),
            "median_is_constant": (
                median_is_constant
            ),
            "median_constant_ace": (
                median_constant
            ),
            "training_ace_median": (
                training_median
            ),
            "median_matches_training": (
                median_matches_training
            ),
            "shuffled_ace_multiset_preserved": (
                shuffled_multiset_preserved
            ),
            "shuffled_mapping_changed": (
                shuffled_mapping_changed
            ),
        }
    )

raw = pd.DataFrame(
    seed_rows
)

raw_path = (
    OUTPUT_ROOT
    / "three_seed_micro_macro_raw.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary_rows = []

for mode in MODES:
    mode_rows = raw[
        raw["mode"] == mode
    ]

    for source, label in METRICS.items():
        for aggregation in (
            "micro",
            "macro",
        ):
            column = (
                f"{aggregation}_{label}"
            )

            summary_rows.append(
                {
                    "mode": mode,
                    "aggregation": (
                        aggregation
                    ),
                    "metric": label,
                    "mean": float(
                        mode_rows[
                            column
                        ].mean()
                    ),
                    "std": float(
                        mode_rows[
                            column
                        ].std(
                            ddof=1
                        )
                    ),
                }
            )

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_ROOT
    / "three_seed_micro_macro_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

effect_rows = []

for seed in SEEDS:
    true_row = raw[
        (raw["seed"] == seed)
        & (raw["mode"] == "true")
    ].iloc[0]

    for mode in (
        "median",
        "shuffled",
    ):
        perturbed_row = raw[
            (raw["seed"] == seed)
            & (raw["mode"] == mode)
        ].iloc[0]

        for _, label in METRICS.items():
            for aggregation in (
                "micro",
                "macro",
            ):
                column = (
                    f"{aggregation}_{label}"
                )

                true_value = float(
                    true_row[column]
                )

                perturbed_value = float(
                    perturbed_row[column]
                )

                delta = (
                    perturbed_value
                    - true_value
                )

                relative_percent = (
                    100.0
                    * delta
                    / true_value
                    if true_value != 0.0
                    else np.nan
                )

                effect_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "aggregation": (
                            aggregation
                        ),
                        "metric": label,
                        "true_value": (
                            true_value
                        ),
                        "perturbed_value": (
                            perturbed_value
                        ),
                        "delta": delta,
                        "relative_change_percent": (
                            relative_percent
                        ),
                    }
                )

effects = pd.DataFrame(
    effect_rows
)

effects_path = (
    OUTPUT_ROOT
    / "three_seed_effects_by_seed.csv"
)

effects.to_csv(
    effects_path,
    index=False,
)

effect_summary = (
    effects
    .groupby(
        [
            "mode",
            "aggregation",
            "metric",
        ],
        as_index=False,
    )
    .agg(
        delta_mean=(
            "delta",
            "mean",
        ),
        delta_std=(
            "delta",
            lambda values: values.std(
                ddof=1
            ),
        ),
        relative_change_percent_mean=(
            "relative_change_percent",
            "mean",
        ),
        relative_change_percent_std=(
            "relative_change_percent",
            lambda values: values.std(
                ddof=1
            ),
        ),
    )
)

effect_summary_path = (
    OUTPUT_ROOT
    / "three_seed_effects_summary.csv"
)

effect_summary.to_csv(
    effect_summary_path,
    index=False,
)

audit_df = pd.DataFrame(
    audit_rows
)

audit_path = (
    OUTPUT_ROOT
    / "ace_perturbation_audit.csv"
)

audit_df.to_csv(
    audit_path,
    index=False,
)

all_protocol_checks_passed = bool(
    audit_df[
        [
            "same_spec_ids_true_median",
            "same_spec_ids_true_shuffled",
            "median_is_constant",
            "median_matches_training",
            "shuffled_ace_multiset_preserved",
            "shuffled_mapping_changed",
        ]
    ]
    .all()
    .all()
)

report = {
    "training_split_resolution": (
        train_resolution
    ),
    "training_ace_median": (
        training_median
    ),
    "all_protocol_checks_passed": (
        all_protocol_checks_passed
    ),
    "experiment_4_complete": (
        all_protocol_checks_passed
    ),
    "artifacts": {
        "raw": str(raw_path),
        "summary": str(
            summary_path
        ),
        "effects_by_seed": str(
            effects_path
        ),
        "effects_summary": str(
            effect_summary_path
        ),
        "audit": str(audit_path),
    },
}

report_path = (
    OUTPUT_ROOT
    / "ace_perturbation_completion.json"
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
print("=" * 96)
print("EXPERIMENT 4: ACE PERTURBATION MICRO + MACRO")
print("=" * 96)
print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 96)
print("PERTURBATION EFFECTS")
print("=" * 96)
print(
    effect_summary.to_string(
        index=False
    )
)

print()
print("=" * 96)
print("PROTOCOL AUDIT")
print("=" * 96)
print(
    audit_df.to_string(
        index=False
    )
)

print()
print("RAW:", raw_path)
print("SUMMARY:", summary_path)
print("EFFECTS:", effect_summary_path)
print("AUDIT:", audit_path)
print("REPORT:", report_path)
print()
print(
    "EXPERIMENT_4_COMPLETE:",
    all_protocol_checks_passed,
)

if not all_protocol_checks_passed:
    print()
    print(
        "实验4尚未通过全部协议检查。"
        "请查看上面的PROTOCOL AUDIT。"
    )
    sys.exit(2)
