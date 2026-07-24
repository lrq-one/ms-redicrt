from __future__ import annotations

import json
from pathlib import Path

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

OUTPUT_ROOT = (
    EXPERIMENT_ROOT
    / "metric_robustness"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS = (
    "cos_raw_0.01",
    "cos_sqrt_0.01",
    "cos_raw_0.05",
    "cos_sqrt_0.05",
    "cos_raw_0.10",
    "cos_sqrt_0.10",
    "jss_0.01",
    "chun_10ppm",
)

AUDIT_COLUMNS = (
    "cos_0.01",
    "cos_raw_0.01",
    "cos_raw_0.01_recomputed_audit",
    "raw_0.01_parity_abs",
)

if not PROC_PATH.is_file():
    raise FileNotFoundError(PROC_PATH)

spec_df = pd.read_pickle(PROC_PATH)

required_mapping_columns = {
    "spec_id",
    "mol_id",
}

missing_mapping = (
    required_mapping_columns
    - set(spec_df.columns)
)

if missing_mapping:
    raise RuntimeError(
        "spec_df缺少字段："
        f"{sorted(missing_mapping)}"
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

seed_rows = []

for seed in (42, 43, 44):
    seed_dir = (
        EXPERIMENT_ROOT
        / f"seed_{seed}"
        / "metric_robustness"
    )

    detail_path = (
        seed_dir
        / "test_per_spectrum_metrics.csv"
    )

    if not detail_path.is_file():
        raise FileNotFoundError(
            detail_path
        )

    detail = pd.read_csv(
        detail_path
    )

    required_columns = {
        "spec_id",
        *METRICS,
        *AUDIT_COLUMNS,
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
            f"seed {seed}测试谱数错误："
            f"{len(detail)}"
        )

    if detail["spec_id"].nunique() != 3931:
        raise RuntimeError(
            f"seed {seed} spec_id不唯一。"
        )

    locked_field_difference = (
        detail["cos_raw_0.01"]
        - detail["cos_0.01"]
    ).abs()

    recomputed_difference = (
        detail[
            "cos_raw_0.01_recomputed_audit"
        ]
        - detail["cos_raw_0.01"]
    ).abs()

    locked_field_max_abs = float(
        locked_field_difference.max()
    )

    renderer_parity_max_abs = float(
        recomputed_difference.max()
    )

    renderer_parity_mean_abs = float(
        recomputed_difference.mean()
    )

    if locked_field_max_abs > 1.0e-12:
        raise RuntimeError(
            f"seed {seed}锁定raw字段不一致："
            f"{locked_field_max_abs}"
        )

    if renderer_parity_max_abs > 2.0e-6:
        raise RuntimeError(
            f"seed {seed} renderer parity失败："
            f"{renderer_parity_max_abs}"
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
            f"seed {seed}有"
            f"{missing_count}条谱无法映射mol_id。"
        )

    per_molecule = (
        merged
        .groupby(
            "mol_id",
            as_index=False,
        )[list(METRICS)]
        .mean()
    )

    if len(per_molecule) != 456:
        raise RuntimeError(
            f"seed {seed}分子数错误："
            f"{len(per_molecule)}"
        )

    per_molecule_path = (
        seed_dir
        / "test_per_molecule_metrics.csv"
    )

    per_molecule.to_csv(
        per_molecule_path,
        index=False,
    )

    result = {
        "seed": seed,
        "spectrum_count": int(
            len(merged)
        ),
        "molecule_count": int(
            len(per_molecule)
        ),
        "locked_raw_field_max_abs": (
            locked_field_max_abs
        ),
        "renderer_parity_max_abs": (
            renderer_parity_max_abs
        ),
        "renderer_parity_mean_abs": (
            renderer_parity_mean_abs
        ),
        "renderer_parity_passed": True,
    }

    for metric in METRICS:
        result[
            f"micro_{metric}"
        ] = float(
            merged[metric].mean()
        )

        result[
            f"macro_{metric}"
        ] = float(
            per_molecule[
                metric
            ].mean()
        )

    result_path = (
        seed_dir
        / "metric_robustness_micro_macro.json"
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

    seed_rows.append(result)

    print()
    print("=" * 80)
    print(f"SEED {seed}")
    print("=" * 80)
    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))

raw = pd.DataFrame(seed_rows)

raw_path = (
    OUTPUT_ROOT
    / "three_seed_raw.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

summary_rows = []

for aggregation in (
    "micro",
    "macro",
):
    for metric in METRICS:
        column = (
            f"{aggregation}_{metric}"
        )

        summary_rows.append(
            {
                "aggregation": aggregation,
                "metric": metric,
                "mean": float(
                    raw[column].mean()
                ),
                "std": float(
                    raw[column].std(
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
    / "three_seed_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

audit = {
    "seed_count": 3,
    "all_renderer_parity_passed": bool(
        raw[
            "renderer_parity_passed"
        ].all()
    ),
    "maximum_renderer_parity_abs": float(
        raw[
            "renderer_parity_max_abs"
        ].max()
    ),
    "maximum_locked_field_abs": float(
        raw[
            "locked_raw_field_max_abs"
        ].max()
    ),
    "protocol": {
        "raw_0.01": (
            "locked output['cos']"
        ),
        "renderer_audit_0.01": (
            "runtime spectrum_allocator "
            "dense renderer"
        ),
        "raw_0.05_0.10": (
            "same runtime dense renderer; "
            "bin_res only changed"
        ),
        "sqrt": (
            "elementwise sqrt after "
            "locked dense rendering"
        ),
        "jss_0.01": (
            "locked output['jss']"
        ),
        "chun_10ppm": (
            "locked Hungarian evaluation"
        ),
    },
}

audit_path = (
    OUTPUT_ROOT
    / "metric_robustness_audit.json"
)

audit_path.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 88)
print(
    "METRIC ROBUSTNESS "
    "THREE-SEED SUMMARY"
)
print("=" * 88)
print(
    summary.to_string(
        index=False
    )
)
print()
print("RAW:", raw_path)
print("SUMMARY:", summary_path)
print("AUDIT:", audit_path)
