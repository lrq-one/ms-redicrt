from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path.cwd()

CURRENT_DIR = (
    ROOT
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
)

OLD_DIR = (
    ROOT
    / ".."
    / "old"
    / "data"
    / "proc"
    / "nist20_full"
).resolve()

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "experiments"
    / "formula_annotation"
    / "ft_hcd_external_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CURRENT_MOL_PATH = (
    CURRENT_DIR
    / "mol_df.pkl"
)

OLD_SPEC_PATH = (
    OLD_DIR
    / "spec_df.pkl"
)

OLD_MOL_PATH = (
    OLD_DIR
    / "mol_df.pkl"
)

OLD_ANN_PATH = (
    OLD_DIR
    / "ann_df.pkl"
)

FUNNEL_PATH = (
    OUTPUT_DIR
    / "selection_funnel.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "ft_hcd_external_summary.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "ft_hcd_external_candidate_manifest.pkl"
)

MANIFEST_CSV_PATH = (
    OUTPUT_DIR
    / "ft_hcd_external_candidate_manifest.csv"
)


def normalize_text(
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


def normalize_id(
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


def sequence_length(
    value: Any,
) -> int:
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


def boolean_mask(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return normalize_text(
        series
    ).isin(
        {
            "TRUE",
            "1",
            "YES",
            "Y",
        }
    )


for path in (
    CURRENT_MOL_PATH,
    OLD_SPEC_PATH,
    OLD_MOL_PATH,
    OLD_ANN_PATH,
):
    if not path.is_file():
        raise FileNotFoundError(
            path.resolve()
        )

print("Loading current molecules...")
current_mol = pd.read_pickle(
    CURRENT_MOL_PATH
)

print("Loading complete NIST20 spectra...")
old_spec = pd.read_pickle(
    OLD_SPEC_PATH
)

print("Loading complete NIST20 molecules...")
old_mol = pd.read_pickle(
    OLD_MOL_PATH
)

print("Loading native annotations...")
old_ann = pd.read_pickle(
    OLD_ANN_PATH
)

current_keys = set(
    connectivity_key(
        current_mol["inchikey_s"]
    )
    .dropna()
    .astype(str)
    .tolist()
)

current_elements: set[str] = set()

for smiles in current_mol[
    "smiles"
].astype(str):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        continue

    current_elements.update(
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    )

spec = old_spec.copy()
ann = old_ann.copy()

for frame in (spec, ann):
    frame["_dset_key"] = (
        normalize_text(
            frame["dset"]
        )
    )

    frame["_dset_spec_id_key"] = (
        normalize_id(
            frame["dset_spec_id"]
        )
    )

    frame["_prec_type_key"] = (
        normalize_text(
            frame["prec_type"]
        )
    )

ann = ann.rename(
    columns={
        "spec_id": (
            "annotation_source_spec_id"
        ),
        "mol_id": (
            "annotation_source_mol_id"
        ),
        "formula": (
            "native_precursor_formula"
        ),
    }
)

ann_columns = [
    "_dset_key",
    "_dset_spec_id_key",
    "_prec_type_key",
    "annotation_source_spec_id",
    "annotation_source_mol_id",
    "native_precursor_formula",
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
]

annotated = spec.merge(
    ann[ann_columns],
    on=[
        "_dset_key",
        "_dset_spec_id_key",
        "_prec_type_key",
    ],
    how="inner",
    validate="one_to_one",
)

if len(annotated) != len(old_ann):
    raise RuntimeError(
        "原生注释与完整谱表连接不完整："
        f"{len(annotated)} vs {len(old_ann)}"
    )

mol_id_mismatch = int(
    (
        annotated["mol_id"]
        .astype(int)
        != annotated[
            "annotation_source_mol_id"
        ].astype(int)
    ).sum()
)

if mol_id_mismatch != 0:
    raise RuntimeError(
        f"注释与谱表mol_id不一致："
        f"{mol_id_mismatch}"
    )

old_mol_meta = old_mol[
    [
        "mol_id",
        "smiles",
        "inchikey_s",
        "formula",
        "exact_mw",
        "single_mol",
        "charge",
        "num_radicals",
    ]
].copy()

old_mol_meta = old_mol_meta.rename(
    columns={
        "formula": (
            "structure_formula"
        )
    }
)

annotated = annotated.merge(
    old_mol_meta,
    on="mol_id",
    how="left",
    validate="many_to_one",
)

annotated[
    "connectivity_key"
] = connectivity_key(
    annotated["inchikey_s"]
)

for column in (
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "col_gas",
    "res",
):
    annotated[
        f"_{column}"
    ] = normalize_text(
        annotated[column]
    )

annotated["ace_numeric"] = (
    pd.to_numeric(
        annotated["ace"],
        errors="coerce",
    )
)

annotated["nce_numeric"] = (
    pd.to_numeric(
        annotated["nce"],
        errors="coerce",
    )
)

annotation_fields = (
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
)

for field in annotation_fields:
    annotated[
        f"{field}_length"
    ] = annotated[field].map(
        sequence_length
    )

annotated[
    "core_lengths_equal"
] = (
    (
        annotated[
            "ann_peak_mzs_length"
        ]
        == annotated[
            "ann_products_length"
        ]
    )
    & (
        annotated[
            "ann_peak_mzs_length"
        ]
        == annotated[
            "ann_exact_mzs_length"
        ]
    )
)

annotated[
    "all_lengths_equal"
] = (
    annotated[
        [
            f"{field}_length"
            for field
            in annotation_fields
        ]
    ]
    .nunique(
        axis=1
    )
    .eq(1)
)

valid_structure_keys: set[str] = set()
invalid_smiles = 0
unsupported_elements = 0

for row in old_mol.itertuples(
    index=False
):
    smiles = str(
        getattr(row, "smiles")
    )

    key = str(
        getattr(row, "inchikey_s")
    ).strip().upper().split("-")[0]

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        invalid_smiles += 1
        continue

    elements = {
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    }

    if not elements.issubset(
        current_elements
    ):
        unsupported_elements += 1
        continue

    valid_structure_keys.add(key)

mh_mask = (
    annotated["_prec_type"]
    == "[M+H]+"
)

ft_mask = (
    annotated["_inst_type"]
    == "FT"
)

hcd_mask = (
    annotated["_frag_mode"]
    == "HCD"
)

ace_mask = (
    annotated["ace_numeric"]
    .notna()
    & np.isfinite(
        annotated["ace_numeric"]
    )
)

nonempty_mask = (
    annotated[
        "ann_peak_mzs_length"
    ].gt(0)
    & annotated[
        "ann_products_length"
    ].gt(0)
    & annotated[
        "ann_exact_mzs_length"
    ].gt(0)
)

single_mask = boolean_mask(
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

valid_structure_mask = (
    annotated[
        "connectivity_key"
    ].isin(
        valid_structure_keys
    )
)

outside_current_mask = (
    ~annotated[
        "connectivity_key"
    ].isin(
        current_keys
    )
)

masks = {
    "all_native_annotations": (
        pd.Series(
            True,
            index=annotated.index,
        )
    ),
    "m_plus_h": (
        mh_mask
    ),
    "m_plus_h_ft_hcd": (
        mh_mask
        & ft_mask
        & hcd_mask
    ),
    "m_plus_h_ft_hcd_ace": (
        mh_mask
        & ft_mask
        & hcd_mask
        & ace_mask
    ),
    "ft_hcd_ace_nonempty_equal_annotations": (
        mh_mask
        & ft_mask
        & hcd_mask
        & ace_mask
        & nonempty_mask
        & annotated[
            "core_lengths_equal"
        ]
    ),
    "ft_hcd_ace_model_compatible": (
        mh_mask
        & ft_mask
        & hcd_mask
        & ace_mask
        & nonempty_mask
        & annotated[
            "core_lengths_equal"
        ]
        & single_mask
        & neutral_mask
        & nonradical_mask
        & valid_structure_mask
    ),
    "ft_hcd_ace_outside_current_connectivities": (
        mh_mask
        & ft_mask
        & hcd_mask
        & ace_mask
        & nonempty_mask
        & annotated[
            "core_lengths_equal"
        ]
        & single_mask
        & neutral_mask
        & nonradical_mask
        & valid_structure_mask
        & outside_current_mask
    ),
}

funnel_rows = []

for stage, mask in masks.items():
    subset = annotated[
        mask
    ]

    funnel_rows.append(
        {
            "stage": stage,
            "spectrum_count": int(
                len(subset)
            ),
            "connectivity_count": int(
                subset[
                    "connectivity_key"
                ].nunique()
            ),
            "ace_nonnull_count": int(
                subset[
                    "ace_numeric"
                ].notna()
                .sum()
            ),
            "nce_nonnull_count": int(
                subset[
                    "nce_numeric"
                ].notna()
                .sum()
            ),
            "core_length_equal_count": int(
                subset[
                    "core_lengths_equal"
                ].sum()
            ),
        }
    )

funnel = pd.DataFrame(
    funnel_rows
)

funnel.to_csv(
    FUNNEL_PATH,
    index=False,
)

manifest = annotated[
    masks[
        "ft_hcd_ace_outside_current_connectivities"
    ]
].copy()

manifest = manifest.sort_values(
    [
        "connectivity_key",
        "ace_numeric",
        "spec_id",
    ]
).reset_index(
    drop=True
)

if manifest[
    "connectivity_key"
].isin(
    current_keys
).any():
    raise RuntimeError(
        "候选manifest仍包含当前2274个结构。"
    )

manifest.to_pickle(
    MANIFEST_PATH
)

csv_columns = [
    "spec_id",
    "dset_spec_id",
    "mol_id",
    "connectivity_key",
    "smiles",
    "structure_formula",
    "native_precursor_formula",
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "col_gas",
    "res",
    "ace_numeric",
    "nce_numeric",
    "prec_mz",
    "ann_peak_mzs_length",
    "ann_products_length",
    "ann_exact_mzs_length",
    "core_lengths_equal",
]

manifest[
    csv_columns
].to_csv(
    MANIFEST_CSV_PATH,
    index=False,
)

spectra_per_molecule = (
    manifest
    .groupby(
        "connectivity_key"
    )
    .size()
    if not manifest.empty
    else pd.Series(
        dtype=float
    )
)

ace_distribution = (
    manifest[
        "ace_numeric"
    ].dropna()
)

nce_distribution = (
    manifest[
        "nce_numeric"
    ].dropna()
)

summary = {
    "status": "audit_complete",
    "important_domain_note": (
        "This candidate benchmark is "
        "FT-HCD and is out-of-domain "
        "relative to the QTOF-CID "
        "training dataset."
    ),
    "spectrum_count": int(
        len(manifest)
    ),
    "connectivity_count": int(
        manifest[
            "connectivity_key"
        ].nunique()
    ),
    "current_connectivity_overlap": int(
        len(
            set(
                manifest[
                    "connectivity_key"
                ].astype(str)
            )
            & current_keys
        )
    ),
    "ace": {
        "nonnull_count": int(
            len(ace_distribution)
        ),
        "unique_count": int(
            ace_distribution.nunique()
        ),
        "min": float(
            ace_distribution.min()
        )
        if len(ace_distribution)
        else None,
        "median": float(
            ace_distribution.median()
        )
        if len(ace_distribution)
        else None,
        "max": float(
            ace_distribution.max()
        )
        if len(ace_distribution)
        else None,
        "q25": float(
            ace_distribution.quantile(
                0.25
            )
        )
        if len(ace_distribution)
        else None,
        "q75": float(
            ace_distribution.quantile(
                0.75
            )
        )
        if len(ace_distribution)
        else None,
    },
    "nce": {
        "nonnull_count": int(
            len(nce_distribution)
        ),
        "unique_count": int(
            nce_distribution.nunique()
        ),
        "min": float(
            nce_distribution.min()
        )
        if len(nce_distribution)
        else None,
        "median": float(
            nce_distribution.median()
        )
        if len(nce_distribution)
        else None,
        "max": float(
            nce_distribution.max()
        )
        if len(nce_distribution)
        else None,
    },
    "spectra_per_connectivity": {
        "min": int(
            spectra_per_molecule.min()
        )
        if len(spectra_per_molecule)
        else None,
        "median": float(
            spectra_per_molecule.median()
        )
        if len(spectra_per_molecule)
        else None,
        "max": int(
            spectra_per_molecule.max()
        )
        if len(spectra_per_molecule)
        else None,
    },
    "protocol_distributions": {
        "resolution": {
            str(key): int(value)
            for key, value in (
                manifest["res"]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        },
        "collision_gas": {
            str(key): int(value)
            for key, value in (
                manifest["col_gas"]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        },
        "ion_mode": {
            str(key): int(value)
            for key, value in (
                manifest["ion_mode"]
                .value_counts(
                    dropna=False
                )
                .items()
            )
        },
    },
    "annotation_field_outer_types": {
        field: {
            str(key): int(value)
            for key, value in Counter(
                type(item).__name__
                for item in manifest[field]
            ).items()
        }
        for field in (
            "ann_peak_mzs",
            "ann_products",
            "ann_exact_mzs",
        )
    },
    "structure_audit": {
        "current_supported_elements": (
            sorted(
                current_elements
            )
        ),
        "invalid_smiles_count": int(
            invalid_smiles
        ),
        "unsupported_element_count": int(
            unsupported_elements
        ),
    },
    "outputs": {
        "funnel": str(
            FUNNEL_PATH.resolve()
        ),
        "manifest_pkl": str(
            MANIFEST_PATH.resolve()
        ),
        "manifest_csv": str(
            MANIFEST_CSV_PATH.resolve()
        ),
    },
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 96)
print("FT-HCD EXTERNAL FORMULA AUDIT")
print("=" * 96)

print(
    funnel.to_string(
        index=False
    )
)

print()
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("FUNNEL:", FUNNEL_PATH.resolve())
print("SUMMARY:", SUMMARY_PATH.resolve())
print("MANIFEST:", MANIFEST_PATH.resolve())
print()
print(
    "FORMULA_FT_HCD_EXTERNAL_AUDIT_COMPLETE"
)
