from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path.cwd()

CURRENT_SPEC_PATH = (
    ROOT
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
    / "spec_df.pkl"
)

CURRENT_MOL_PATH = (
    ROOT
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
    / "mol_df.pkl"
)

OLD_DIR = (
    ROOT
    / ".."
    / "old"
    / "data"
    / "proc"
    / "nist20_full"
).resolve()

OLD_SPEC_PATH = OLD_DIR / "spec_df.pkl"
OLD_MOL_PATH = OLD_DIR / "mol_df.pkl"
OLD_ANN_PATH = OLD_DIR / "ann_df.pkl"

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "experiments"
    / "formula_annotation"
    / "protocol_bottleneck"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "protocol_filter_counts.csv"
)


def norm_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )


def norm_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"^(-?\d+)\.0$",
            r"\1",
            regex=True,
        )
    )


def connectivity_key(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
        .str.split("-")
        .str[0]
    )


def seq_len(value: Any) -> int:
    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
            pd.Series,
        ),
    ):
        return int(len(value))

    return 0


def bool_mask(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        norm_text(series)
        .isin(
            {
                "TRUE",
                "1",
                "YES",
                "Y",
            }
        )
    )


for path in (
    CURRENT_SPEC_PATH,
    CURRENT_MOL_PATH,
    OLD_SPEC_PATH,
    OLD_MOL_PATH,
    OLD_ANN_PATH,
):
    if not path.is_file():
        raise FileNotFoundError(
            path.resolve()
        )

print("Loading tables...")

current_spec = pd.read_pickle(
    CURRENT_SPEC_PATH
)

current_mol = pd.read_pickle(
    CURRENT_MOL_PATH
)

old_spec = pd.read_pickle(
    OLD_SPEC_PATH
)

old_mol = pd.read_pickle(
    OLD_MOL_PATH
)

old_ann = pd.read_pickle(
    OLD_ANN_PATH
)

spec = old_spec.copy()
ann = old_ann.copy()

# 注释表与完整谱表的稳定连接键。
for frame in (spec, ann):
    frame["_dset"] = norm_text(
        frame["dset"]
    )

    frame["_dset_spec_id"] = norm_id(
        frame["dset_spec_id"]
    )

    frame["_prec_type"] = norm_text(
        frame["prec_type"]
    )

ann_columns = [
    "_dset",
    "_dset_spec_id",
    "_prec_type",
    "formula",
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
]

annotated = spec.merge(
    ann[ann_columns],
    on=[
        "_dset",
        "_dset_spec_id",
        "_prec_type",
    ],
    how="inner",
    validate="one_to_one",
)

if len(annotated) != len(old_ann):
    raise RuntimeError(
        "old_ann与old_spec连接不完整："
        f"{len(annotated)} vs {len(old_ann)}"
    )

annotated = annotated.merge(
    old_mol[
        [
            "mol_id",
            "inchikey_s",
            "smiles",
            "single_mol",
            "charge",
            "num_radicals",
        ]
    ],
    on="mol_id",
    how="left",
    validate="many_to_one",
)

annotated["_connectivity"] = (
    connectivity_key(
        annotated["inchikey_s"]
    )
)

current_connectivities = set(
    connectivity_key(
        current_mol["inchikey_s"]
    )
    .dropna()
    .astype(str)
    .tolist()
)

# 关键修复：prec_type也必须在current_spec中生成标准化列。
protocol_columns = (
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "col_gas",
    "res",
)

for column in protocol_columns:
    annotated[
        f"_{column}"
    ] = norm_text(
        annotated[column]
    )

    current_spec[
        f"_{column}"
    ] = norm_text(
        current_spec[column]
    )

annotated["_ace_numeric"] = (
    pd.to_numeric(
        annotated["ace"],
        errors="coerce",
    )
)

annotated["_ann_peak_len"] = (
    annotated[
        "ann_peak_mzs"
    ].map(seq_len)
)

annotated["_ann_product_len"] = (
    annotated[
        "ann_products"
    ].map(seq_len)
)

annotated["_ann_exact_len"] = (
    annotated[
        "ann_exact_mzs"
    ].map(seq_len)
)

annotated["_core_lengths_equal"] = (
    (
        annotated[
            "_ann_peak_len"
        ]
        == annotated[
            "_ann_product_len"
        ]
    )
    & (
        annotated[
            "_ann_peak_len"
        ]
        == annotated[
            "_ann_exact_len"
        ]
    )
)

print()
print("=" * 96)
print("CURRENT SAFE PROTOCOL VALUES")
print("=" * 96)

for column in (
    "_prec_type",
    "_inst_type",
    "_frag_mode",
    "_spec_type",
    "_ion_mode",
    "_col_gas",
    "_res",
):
    print()
    print(column)

    print(
        current_spec[column]
        .value_counts(
            dropna=False
        )
        .head(30)
        .to_string()
    )

print()
print("=" * 96)
print("NATIVE ANNOTATION PROTOCOL VALUES")
print("=" * 96)

for column in (
    "_prec_type",
    "_inst_type",
    "_frag_mode",
    "_spec_type",
    "_ion_mode",
    "_col_gas",
    "_res",
):
    print()
    print(column)

    print(
        annotated[column]
        .value_counts(
            dropna=False
        )
        .head(30)
        .to_string()
    )

current_inst_values = set(
    current_spec["_inst_type"]
    .dropna()
    .astype(str)
    .tolist()
)

current_frag_values = set(
    current_spec["_frag_mode"]
    .dropna()
    .astype(str)
    .tolist()
)

print()
print(
    "Current instrument labels:",
    sorted(current_inst_values),
)

print(
    "Current fragmentation labels:",
    sorted(current_frag_values),
)

mh_mask = (
    annotated["_prec_type"]
    == "[M+H]+"
)

# 广义字符串筛选。
qtof_mask = (
    annotated["_inst_type"]
    .str.contains(
        r"Q-?TOF|QTOF",
        regex=True,
        na=False,
    )
)

orbitrap_mask = (
    annotated["_inst_type"]
    .str.contains(
        "ORBITRAP",
        regex=False,
        na=False,
    )
)

cid_mask = (
    annotated["_frag_mode"]
    .str.contains(
        "CID",
        regex=False,
        na=False,
    )
)

hcd_mask = (
    annotated["_frag_mode"]
    .str.contains(
        "HCD",
        regex=False,
        na=False,
    )
)

# 与当前数据实际标签完全一致的筛选。
current_inst_mask = (
    annotated["_inst_type"]
    .isin(current_inst_values)
)

current_frag_mask = (
    annotated["_frag_mode"]
    .isin(current_frag_values)
)

ace_mask = (
    annotated["_ace_numeric"]
    .notna()
    & np.isfinite(
        annotated["_ace_numeric"]
    )
)

nonempty_core_mask = (
    annotated["_ann_peak_len"].gt(0)
    & annotated[
        "_ann_product_len"
    ].gt(0)
    & annotated[
        "_ann_exact_len"
    ].gt(0)
)

outside_current_mask = (
    ~annotated["_connectivity"]
    .isin(current_connectivities)
)

single_mask = bool_mask(
    annotated["single_mol"]
)

neutral_mask = (
    pd.to_numeric(
        annotated["charge"],
        errors="coerce",
    )
    .fillna(0)
    .eq(0)
)

nonradical_mask = (
    pd.to_numeric(
        annotated["num_radicals"],
        errors="coerce",
    )
    .fillna(0)
    .eq(0)
)

model_compatible_mask = (
    single_mask
    & neutral_mask
    & nonradical_mask
)

current_exact_mask = mh_mask.copy()

for column in (
    "_inst_type",
    "_frag_mode",
    "_spec_type",
    "_ion_mode",
    "_col_gas",
    "_res",
):
    allowed = set(
        current_spec[column]
        .dropna()
        .astype(str)
        .tolist()
    )

    current_exact_mask &= (
        annotated[column]
        .isin(allowed)
    )

rows: list[dict[str, Any]] = []


def add(
    name: str,
    mask: pd.Series,
) -> None:
    subset = annotated[mask]

    rows.append(
        {
            "stage": name,
            "spectrum_count": int(
                len(subset)
            ),
            "molecule_count": int(
                subset[
                    "_connectivity"
                ].nunique()
            ),
            "ace_nonnull_count": int(
                subset[
                    "_ace_numeric"
                ].notna()
                .sum()
            ),
            "nonempty_core_annotation_count": int(
                (
                    subset[
                        "_ann_peak_len"
                    ].gt(0)
                    & subset[
                        "_ann_product_len"
                    ].gt(0)
                    & subset[
                        "_ann_exact_len"
                    ].gt(0)
                ).sum()
            ),
            "core_length_equal_count": int(
                subset[
                    "_core_lengths_equal"
                ].sum()
            ),
        }
    )


all_mask = pd.Series(
    True,
    index=annotated.index,
)

add(
    "all_native_annotations",
    all_mask,
)

add(
    "m_plus_h",
    mh_mask,
)

add(
    "m_plus_h_qtof_any_fragmentation",
    mh_mask & qtof_mask,
)

add(
    "m_plus_h_any_instrument_cid",
    mh_mask & cid_mask,
)

add(
    "m_plus_h_qtof_cid",
    mh_mask
    & qtof_mask
    & cid_mask,
)

add(
    "m_plus_h_qtof_cid_ace",
    mh_mask
    & qtof_mask
    & cid_mask
    & ace_mask,
)

add(
    "m_plus_h_current_instrument",
    mh_mask
    & current_inst_mask,
)

add(
    "m_plus_h_current_fragmentation",
    mh_mask
    & current_frag_mask,
)

add(
    "m_plus_h_current_instrument_and_fragmentation",
    mh_mask
    & current_inst_mask
    & current_frag_mask,
)

add(
    "m_plus_h_current_instrument_fragmentation_ace",
    mh_mask
    & current_inst_mask
    & current_frag_mask
    & ace_mask,
)

add(
    "m_plus_h_orbitrap_hcd",
    mh_mask
    & orbitrap_mask
    & hcd_mask,
)

add(
    "m_plus_h_orbitrap_hcd_ace",
    mh_mask
    & orbitrap_mask
    & hcd_mask
    & ace_mask,
)

add(
    "exact_current_metadata_protocol",
    current_exact_mask,
)

add(
    "exact_current_protocol_with_ace",
    current_exact_mask
    & ace_mask,
)

add(
    "qtof_cid_ace_nonempty_annotations",
    mh_mask
    & qtof_mask
    & cid_mask
    & ace_mask
    & nonempty_core_mask,
)

add(
    "qtof_cid_ace_model_compatible",
    mh_mask
    & qtof_mask
    & cid_mask
    & ace_mask
    & nonempty_core_mask
    & model_compatible_mask,
)

add(
    "qtof_cid_outside_current_connectivities",
    mh_mask
    & qtof_mask
    & cid_mask
    & ace_mask
    & nonempty_core_mask
    & model_compatible_mask
    & outside_current_mask,
)

add(
    "current_domain_outside_current_connectivities",
    mh_mask
    & current_inst_mask
    & current_frag_mask
    & ace_mask
    & nonempty_core_mask
    & model_compatible_mask
    & outside_current_mask,
)

add(
    "orbitrap_hcd_outside_current_connectivities",
    mh_mask
    & orbitrap_mask
    & hcd_mask
    & ace_mask
    & nonempty_core_mask
    & model_compatible_mask
    & outside_current_mask,
)

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 96)
print("FILTER BOTTLENECK COUNTS")
print("=" * 96)

print(
    result.to_string(
        index=False
    )
)

print()
print("WROTE:", OUTPUT_PATH.resolve())
print()
print(
    "FORMULA_PROTOCOL_BOTTLENECK_AUDIT_COMPLETE"
)
