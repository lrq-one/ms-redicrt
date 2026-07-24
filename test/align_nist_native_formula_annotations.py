from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd()

SAFE_DIR = (
    ROOT
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
)

SAFE_SPEC_PATH = (
    SAFE_DIR
    / "spec_df.pkl"
)

OLD_ANN_CANDIDATES = (
    (
        ROOT
        / ".."
        / "old"
        / "data"
        / "proc"
        / "nist20_full"
        / "ann_df.pkl"
    ).resolve(),
    (
        ROOT.parent
        / "old"
        / "data"
        / "proc"
        / "nist20_full"
        / "ann_df.pkl"
    ).resolve(),
)

RANDOM_SPLIT_ROOT = (
    ROOT
    / "data"
    / "split"
    / "nist20_qtof_cid_safe19659_qcv1_trainonly"
)

SCAFFOLD_SPLIT_ROOT = (
    ROOT
    / "data"
    / "split"
    / "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42"
)

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "experiments"
    / "formula_annotation"
    / "native_alignment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALIGNED_PATH = (
    SAFE_DIR
    / "ann_df_nist_native_aligned.pkl"
)

ANNOTATED_ONLY_PATH = (
    SAFE_DIR
    / "ann_df_nist_native_annotated_only.pkl"
)

COVERAGE_PATH = (
    OUTPUT_DIR
    / "annotation_coverage_summary.csv"
)

SCHEMA_AUDIT_PATH = (
    OUTPUT_DIR
    / "annotation_schema_audit.json"
)

AMBIGUITY_PATH = (
    OUTPUT_DIR
    / "ambiguous_annotation_keys.csv"
)

SAMPLE_PATH = (
    OUTPUT_DIR
    / "annotation_samples.txt"
)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            path.resolve()
        )

    return path.resolve()


def find_old_annotation_file() -> Path:
    for path in OLD_ANN_CANDIDATES:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "没有找到旧NIST20注释文件。已检查：\n"
        + "\n".join(
            str(path)
            for path in OLD_ANN_CANDIDATES
        )
    )


def normalize_string(
    series: pd.Series,
    lower: bool = False,
) -> pd.Series:
    result = (
        series
        .astype("string")
        .str.strip()
    )

    result = result.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )

    if lower:
        result = result.str.lower()

    return result


def normalize_identifier(
    series: pd.Series,
) -> pd.Series:
    result = normalize_string(
        series,
        lower=False,
    )

    # 避免CSV或pickle中的整数ID被显示成123.0。
    result = result.str.replace(
        r"^(-?\d+)\.0$",
        r"\1",
        regex=True,
    )

    return result


def normalize_precursor_type(
    series: pd.Series,
) -> pd.Series:
    return (
        normalize_string(
            series,
            lower=True,
        )
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )


def is_sequence(value: Any) -> bool:
    return isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
            pd.Series,
        ),
    )


def sequence_length(value: Any) -> int | None:
    if not is_sequence(value):
        return None

    try:
        return int(len(value))
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        if (
            isinstance(value, float)
            and not math.isfinite(value)
        ):
            return None

        return value

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)


def series_type_counts(
    series: pd.Series,
) -> dict[str, int]:
    counts = Counter(
        type(value).__name__
        for value in series
    )

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def sequence_field_audit(
    series: pd.Series,
) -> dict[str, Any]:
    sequence_mask = series.map(
        is_sequence
    )

    lengths = series.map(
        sequence_length
    )

    valid_lengths = pd.to_numeric(
        lengths,
        errors="coerce",
    ).dropna()

    nonempty = (
        valid_lengths > 0
    )

    flattened_type_counts: Counter = Counter()
    flattened_examples: list[str] = []

    for value in series[
        sequence_mask
    ].head(2000):
        for item in list(value)[:20]:
            flattened_type_counts[
                type(item).__name__
            ] += 1

            if len(flattened_examples) < 20:
                flattened_examples.append(
                    repr(item)[:200]
                )

    if valid_lengths.empty:
        minimum = None
        maximum = None
        mean = None
        median = None
    else:
        minimum = int(
            valid_lengths.min()
        )
        maximum = int(
            valid_lengths.max()
        )
        mean = float(
            valid_lengths.mean()
        )
        median = float(
            valid_lengths.median()
        )

    return {
        "row_count": int(
            len(series)
        ),
        "python_type_counts": (
            series_type_counts(series)
        ),
        "sequence_row_count": int(
            sequence_mask.sum()
        ),
        "non_sequence_row_count": int(
            (~sequence_mask).sum()
        ),
        "nonempty_sequence_count": int(
            nonempty.sum()
        ),
        "empty_sequence_count": int(
            (valid_lengths == 0).sum()
        ),
        "length_min": minimum,
        "length_max": maximum,
        "length_mean": mean,
        "length_median": median,
        "flattened_entry_type_counts": {
            str(key): int(value)
            for key, value
            in flattened_type_counts.items()
        },
        "flattened_entry_examples": (
            flattened_examples
        ),
    }


def read_split_ids(path: Path) -> set[int]:
    path = require_file(path)

    frame = pd.read_csv(path)

    frame = frame.drop(
        columns=[
            column
            for column in frame.columns
            if str(column).lower().startswith(
                "unnamed:"
            )
        ],
        errors="ignore",
    )

    if frame.empty:
        return set()

    column_map = {
        str(column).lower(): column
        for column in frame.columns
    }

    if "spec_id" in column_map:
        column = column_map[
            "spec_id"
        ]
    else:
        column = frame.columns[0]

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    return set(
        values.astype(int).tolist()
    )


def annotation_nonempty_mask(
    frame: pd.DataFrame,
) -> pd.Series:
    peak_lengths = frame[
        "ann_peak_mzs"
    ].map(
        sequence_length
    )

    product_lengths = frame[
        "ann_products"
    ].map(
        sequence_length
    )

    return (
        frame[
            "annotation_row_found"
        ].fillna(False).astype(bool)
        & peak_lengths.fillna(0).gt(0)
        & product_lengths.fillna(0).gt(0)
    )


def make_coverage_row(
    name: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target_count = int(
        len(frame)
    )

    mapped_count = int(
        frame[
            "annotation_row_found"
        ]
        .fillna(False)
        .sum()
    )

    annotated_mask = (
        annotation_nonempty_mask(
            frame
        )
    )

    annotated_count = int(
        annotated_mask.sum()
    )

    return {
        "dataset_split": name,
        "spectrum_count": target_count,
        "molecule_count": int(
            frame["mol_id"].nunique()
        ),
        "annotation_row_found_count": (
            mapped_count
        ),
        "annotation_row_found_percent": (
            100.0
            * mapped_count
            / target_count
            if target_count
            else np.nan
        ),
        "nonempty_product_annotation_count": (
            annotated_count
        ),
        "nonempty_product_annotation_percent": (
            100.0
            * annotated_count
            / target_count
            if target_count
            else np.nan
        ),
        "annotated_molecule_count": int(
            frame.loc[
                annotated_mask,
                "mol_id",
            ].nunique()
        ),
    }


safe_spec_path = require_file(
    SAFE_SPEC_PATH
)

old_ann_path = find_old_annotation_file()

print()
print("=" * 96)
print("EXPERIMENT 5B-1: NIST NATIVE ANNOTATION ALIGNMENT")
print("=" * 96)
print("Current safe spec:", safe_spec_path)
print("Old native annotations:", old_ann_path)
print()

safe = pd.read_pickle(
    safe_spec_path
)

ann = pd.read_pickle(
    old_ann_path
)

print(
    "safe spec shape:",
    safe.shape,
)

print(
    "native ann shape:",
    ann.shape,
)

print(
    "safe columns:",
    list(safe.columns),
)

print(
    "annotation columns:",
    list(ann.columns),
)

required_safe_columns = {
    "spec_id",
    "mol_id",
    "dset",
    "dset_spec_id",
    "prec_type",
}

required_ann_columns = {
    "dset",
    "dset_spec_id",
    "prec_type",
    "formula",
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
}

missing_safe = (
    required_safe_columns
    - set(safe.columns)
)

missing_ann = (
    required_ann_columns
    - set(ann.columns)
)

if missing_safe:
    raise RuntimeError(
        "safe spec缺少字段："
        f"{sorted(missing_safe)}"
    )

if missing_ann:
    raise RuntimeError(
        "native annotation缺少字段："
        f"{sorted(missing_ann)}"
    )

safe = safe.copy()
ann = ann.copy()

safe["_dset_key"] = normalize_string(
    safe["dset"],
    lower=True,
)

ann["_dset_key"] = normalize_string(
    ann["dset"],
    lower=True,
)

safe["_dset_spec_id_key"] = (
    normalize_identifier(
        safe["dset_spec_id"]
    )
)

ann["_dset_spec_id_key"] = (
    normalize_identifier(
        ann["dset_spec_id"]
    )
)

safe["_prec_type_key"] = (
    normalize_precursor_type(
        safe["prec_type"]
    )
)

ann["_prec_type_key"] = (
    normalize_precursor_type(
        ann["prec_type"]
    )
)

candidate_key_sets = [
    [
        "_dset_key",
        "_dset_spec_id_key",
    ],
    [
        "_dset_key",
        "_dset_spec_id_key",
        "_prec_type_key",
    ],
]

chosen_keys: list[str] | None = None
key_audits: list[dict[str, Any]] = []

for keys in candidate_key_sets:
    safe_keys = (
        safe[keys]
        .drop_duplicates()
    )

    relevant_ann = ann.merge(
        safe_keys,
        on=keys,
        how="inner",
    )

    safe_duplicate_rows = int(
        safe.duplicated(
            subset=keys,
            keep=False,
        ).sum()
    )

    relevant_ann_duplicate_rows = int(
        relevant_ann.duplicated(
            subset=keys,
            keep=False,
        ).sum()
    )

    audit = {
        "keys": keys,
        "safe_unique_key_count": int(
            len(safe_keys)
        ),
        "safe_duplicate_row_count": (
            safe_duplicate_rows
        ),
        "relevant_annotation_row_count": int(
            len(relevant_ann)
        ),
        "relevant_annotation_duplicate_row_count": (
            relevant_ann_duplicate_rows
        ),
    }

    key_audits.append(audit)

    if (
        safe_duplicate_rows == 0
        and relevant_ann_duplicate_rows == 0
    ):
        chosen_keys = keys
        break

print()
print("JOIN KEY AUDIT")
print(
    json.dumps(
        key_audits,
        indent=2,
        ensure_ascii=False,
    )
)

if chosen_keys is None:
    base_keys = [
        "_dset_key",
        "_dset_spec_id_key",
    ]

    safe_keys = (
        safe[base_keys]
        .drop_duplicates()
    )

    relevant_ann = ann.merge(
        safe_keys,
        on=base_keys,
        how="inner",
    )

    ambiguous = relevant_ann[
        relevant_ann.duplicated(
            subset=base_keys,
            keep=False,
        )
    ].copy()

    keep_columns = [
        column
        for column in (
            "dset",
            "dset_spec_id",
            "prec_type",
            "spec_id",
            "mol_id",
            "formula",
            *base_keys,
        )
        if column in ambiguous.columns
    ]

    ambiguous[
        keep_columns
    ].to_csv(
        AMBIGUITY_PATH,
        index=False,
    )

    print()
    print(
        "无法获得唯一注释映射。"
    )
    print(
        "歧义记录已写入：",
        AMBIGUITY_PATH,
    )

    sys.exit(2)

print()
print(
    "CHOSEN JOIN KEYS:",
    chosen_keys,
)

safe_for_merge = safe[
    [
        "spec_id",
        "mol_id",
        "dset",
        "dset_spec_id",
        "prec_type",
        *chosen_keys,
    ]
].copy()

safe_for_merge = safe_for_merge.rename(
    columns={
        "spec_id": "spec_id",
        "mol_id": "mol_id",
        "dset": "current_dset",
        "dset_spec_id": (
            "current_dset_spec_id"
        ),
        "prec_type": (
            "current_prec_type"
        ),
    }
)

ann_columns = [
    *chosen_keys,
    "formula",
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
]

for optional_column in (
    "spec_id",
    "mol_id",
    "dset",
    "dset_spec_id",
    "prec_type",
):
    if optional_column in ann.columns:
        ann_columns.append(
            optional_column
        )

ann_for_merge = ann[
    list(
        dict.fromkeys(
            ann_columns
        )
    )
].copy()

ann_for_merge = ann_for_merge.rename(
    columns={
        "formula": (
            "native_precursor_formula"
        ),
        "spec_id": (
            "native_source_spec_id"
        ),
        "mol_id": (
            "native_source_mol_id"
        ),
        "dset": (
            "native_source_dset"
        ),
        "dset_spec_id": (
            "native_source_dset_spec_id"
        ),
        "prec_type": (
            "native_source_prec_type"
        ),
    }
)

aligned = safe_for_merge.merge(
    ann_for_merge,
    on=chosen_keys,
    how="left",
    validate="one_to_one",
    indicator=True,
)

aligned[
    "annotation_row_found"
] = (
    aligned["_merge"] == "both"
)

aligned = aligned.drop(
    columns=[
        "_merge",
        *chosen_keys,
    ],
    errors="ignore",
)

aligned[
    "has_nonempty_product_annotation"
] = annotation_nonempty_mask(
    aligned
)

aligned = aligned.sort_values(
    "spec_id"
).reset_index(
    drop=True
)

if len(aligned) != len(safe):
    raise RuntimeError(
        "对齐后行数发生变化："
        f"safe={len(safe)}, "
        f"aligned={len(aligned)}"
    )

if aligned["spec_id"].nunique() != len(
    aligned
):
    raise RuntimeError(
        "对齐后spec_id不唯一。"
    )

coverage_rows = [
    make_coverage_row(
        "safe19659_all",
        aligned,
    )
]

split_definitions = {
    "random_train": (
        RANDOM_SPLIT_ROOT
        / "train_ids.csv"
    ),
    "random_val": (
        RANDOM_SPLIT_ROOT
        / "val_ids.csv"
    ),
    "random_test": (
        RANDOM_SPLIT_ROOT
        / "test_ids.csv"
    ),
    "scaffold_train": (
        SCAFFOLD_SPLIT_ROOT
        / "train_ids.csv"
    ),
    "scaffold_val": (
        SCAFFOLD_SPLIT_ROOT
        / "val_ids.csv"
    ),
    "scaffold_test": (
        SCAFFOLD_SPLIT_ROOT
        / "test_ids.csv"
    ),
}

split_id_audit = {}

for split_name, split_path in (
    split_definitions.items()
):
    if not split_path.is_file():
        split_id_audit[
            split_name
        ] = {
            "status": "missing",
            "path": str(
                split_path.resolve()
            ),
        }
        continue

    ids = read_split_ids(
        split_path
    )

    subset = aligned[
        aligned["spec_id"].isin(ids)
    ].copy()

    split_id_audit[
        split_name
    ] = {
        "status": "loaded",
        "path": str(
            split_path.resolve()
        ),
        "id_count": int(
            len(ids)
        ),
        "matched_current_spectra": int(
            len(subset)
        ),
    }

    coverage_rows.append(
        make_coverage_row(
            split_name,
            subset,
        )
    )

coverage = pd.DataFrame(
    coverage_rows
)

sequence_fields = (
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
)

mapped = aligned[
    aligned[
        "annotation_row_found"
    ]
].copy()

sequence_audits = {
    field: sequence_field_audit(
        mapped[field]
    )
    for field in sequence_fields
}

length_frame = pd.DataFrame(
    {
        field: mapped[field].map(
            sequence_length
        )
        for field in sequence_fields
    },
    index=mapped.index,
)

all_lengths_available = (
    length_frame.notna().all(
        axis=1
    )
)

comparable_lengths = length_frame[
    all_lengths_available
].copy()

if comparable_lengths.empty:
    equal_all_five_count = 0
    unequal_all_five_count = 0
else:
    equal_all_five = (
        comparable_lengths.nunique(
            axis=1
        )
        == 1
    )

    equal_all_five_count = int(
        equal_all_five.sum()
    )

    unequal_all_five_count = int(
        (~equal_all_five).sum()
    )

pairwise_length_checks = {}

for left, right in (
    (
        "ann_peak_mzs",
        "ann_products",
    ),
    (
        "ann_peak_mzs",
        "ann_losses",
    ),
    (
        "ann_peak_mzs",
        "ann_isotopes",
    ),
    (
        "ann_peak_mzs",
        "ann_exact_mzs",
    ),
    (
        "ann_products",
        "ann_exact_mzs",
    ),
):
    valid = (
        length_frame[left].notna()
        & length_frame[right].notna()
    )

    equal = (
        length_frame.loc[
            valid,
            left,
        ]
        == length_frame.loc[
            valid,
            right,
        ]
    )

    pairwise_length_checks[
        f"{left}__vs__{right}"
    ] = {
        "comparable_row_count": int(
            valid.sum()
        ),
        "equal_length_count": int(
            equal.sum()
        ),
        "unequal_length_count": int(
            (~equal).sum()
        ),
        "equal_length_percent": float(
            100.0
            * equal.sum()
            / len(equal)
        )
        if len(equal)
        else None,
    }

duplicate_current_key_count = int(
    safe.duplicated(
        subset=chosen_keys,
        keep=False,
    ).sum()
)

relevant_ann_keys = ann.merge(
    safe[chosen_keys].drop_duplicates(),
    on=chosen_keys,
    how="inner",
)

duplicate_native_key_count = int(
    relevant_ann_keys.duplicated(
        subset=chosen_keys,
        keep=False,
    ).sum()
)

schema_audit = {
    "status": "complete",
    "paths": {
        "safe_spec": str(
            safe_spec_path
        ),
        "old_native_annotations": str(
            old_ann_path
        ),
        "aligned_output": str(
            ALIGNED_PATH.resolve()
        ),
        "annotated_only_output": str(
            ANNOTATED_ONLY_PATH.resolve()
        ),
        "coverage_output": str(
            COVERAGE_PATH.resolve()
        ),
    },
    "shapes": {
        "safe_spec": [
            int(value)
            for value in safe.shape
        ],
        "old_native_annotations": [
            int(value)
            for value in ann.shape
        ],
        "aligned": [
            int(value)
            for value in aligned.shape
        ],
        "mapped": [
            int(value)
            for value in mapped.shape
        ],
    },
    "join": {
        "candidate_key_audits": (
            key_audits
        ),
        "chosen_keys": chosen_keys,
        "duplicate_current_key_row_count": (
            duplicate_current_key_count
        ),
        "duplicate_relevant_native_key_row_count": (
            duplicate_native_key_count
        ),
        "mapped_row_count": int(
            aligned[
                "annotation_row_found"
            ].sum()
        ),
        "unmapped_row_count": int(
            (
                ~aligned[
                    "annotation_row_found"
                ]
            ).sum()
        ),
    },
    "split_id_audit": (
        split_id_audit
    ),
    "sequence_field_audits": (
        sequence_audits
    ),
    "length_consistency": {
        "all_five_fields_comparable_count": int(
            len(comparable_lengths)
        ),
        "all_five_fields_equal_length_count": (
            equal_all_five_count
        ),
        "all_five_fields_unequal_length_count": (
            unequal_all_five_count
        ),
        "pairwise": (
            pairwise_length_checks
        ),
    },
    "formula_type_counts": (
        series_type_counts(
            mapped[
                "native_precursor_formula"
            ]
        )
    ),
}

aligned.to_pickle(
    ALIGNED_PATH
)

annotated_only = aligned[
    aligned[
        "has_nonempty_product_annotation"
    ]
].copy()

annotated_only.to_pickle(
    ANNOTATED_ONLY_PATH
)

coverage.to_csv(
    COVERAGE_PATH,
    index=False,
)

SCHEMA_AUDIT_PATH.write_text(
    json.dumps(
        json_safe(
            schema_audit
        ),
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

sample_columns = [
    "spec_id",
    "mol_id",
    "current_dset",
    "current_dset_spec_id",
    "current_prec_type",
    "native_precursor_formula",
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
]

with SAMPLE_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    sample_frame = annotated_only[
        sample_columns
    ].head(10)

    for row_number, (
        _,
        row,
    ) in enumerate(
        sample_frame.iterrows(),
        start=1,
    ):
        handle.write(
            "=" * 100
            + "\n"
        )
        handle.write(
            f"SAMPLE {row_number}\n"
        )
        handle.write(
            "=" * 100
            + "\n"
        )

        for column in sample_columns:
            value = row[column]

            handle.write(
                f"{column}:\n"
            )
            handle.write(
                repr(value)[:5000]
                + "\n\n"
            )

print()
print("=" * 96)
print("ALIGNMENT COVERAGE")
print("=" * 96)
print(
    coverage.to_string(
        index=False
    )
)

print()
print("=" * 96)
print("LENGTH CONSISTENCY")
print("=" * 96)
print(
    json.dumps(
        json_safe(
            schema_audit[
                "length_consistency"
            ]
        ),
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("=" * 96)
print("OUTPUTS")
print("=" * 96)
print("Aligned all:", ALIGNED_PATH.resolve())
print(
    "Annotated only:",
    ANNOTATED_ONLY_PATH.resolve(),
)
print("Coverage:", COVERAGE_PATH.resolve())
print(
    "Schema audit:",
    SCHEMA_AUDIT_PATH.resolve(),
)
print("Samples:", SAMPLE_PATH.resolve())

print()
print(
    "MAPPED ROWS:",
    int(
        aligned[
            "annotation_row_found"
        ].sum()
    ),
    "/",
    len(aligned),
)

print(
    "NONEMPTY PRODUCT ANNOTATIONS:",
    int(
        aligned[
            "has_nonempty_product_annotation"
        ].sum()
    ),
    "/",
    len(aligned),
)

print()
print("EXPERIMENT_5B_ALIGNMENT_COMPLETE")
