from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


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

RANDOM_SPLIT_DIR = (
    ROOT
    / "data"
    / "split"
    / "nist20_qtof_cid_safe19659_qcv1_trainonly"
)

SCAFFOLD_SPLIT_DIR = (
    ROOT
    / "data"
    / "split"
    / "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42"
)

OUTPUT_DIR = (
    ROOT
    / "runs"
    / "experiments"
    / "experiment5_source_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PATHS = {
    "current_spec": (
        CURRENT_DIR / "spec_df.pkl"
    ),
    "current_mol": (
        CURRENT_DIR / "mol_df.pkl"
    ),
    "old_spec": (
        OLD_DIR / "spec_df.pkl"
    ),
    "old_mol": (
        OLD_DIR / "mol_df.pkl"
    ),
    "old_ann": (
        OLD_DIR / "ann_df.pkl"
    ),
}

RETRIEVAL_DETAIL_PATH = (
    OUTPUT_DIR
    / "retrieval_target_coverage.csv"
)

RETRIEVAL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "retrieval_coverage_summary.csv"
)

RETRIEVAL_LIBRARY_PATH = (
    OUTPUT_DIR
    / "retrieval_library_audit.json"
)

FORMULA_SUMMARY_PATH = (
    OUTPUT_DIR
    / "formula_benchmark_summary.csv"
)

FORMULA_SCHEMA_PATH = (
    OUTPUT_DIR
    / "formula_annotation_schema_audit.json"
)

FORMULA_MANIFEST_PATH = (
    OUTPUT_DIR
    / "formula_benchmark_candidate_manifest.pkl"
)

FORMULA_MANIFEST_CSV_PATH = (
    OUTPUT_DIR
    / "formula_benchmark_candidate_manifest.csv"
)

FINAL_REPORT_PATH = (
    OUTPUT_DIR
    / "experiment5_source_audit.json"
)

PPM = 10.0
NEGATIVE_COUNT_REQUIRED = 49
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            path.resolve()
        )

    return path.resolve()


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


def normalize_id(
    series: pd.Series,
) -> pd.Series:
    return (
        normalize_text(series)
        .str.replace(
            r"^(-?\d+)\.0$",
            r"\1",
            regex=True,
        )
    )


def normalize_protocol(
    series: pd.Series,
) -> pd.Series:
    return (
        normalize_text(series)
        .str.lower()
        .str.replace(
            " ",
            "",
            regex=False,
        )
    )


def connectivity_key(
    series: pd.Series,
) -> pd.Series:
    return (
        normalize_text(series)
        .str.upper()
        .str.split("-")
        .str[0]
    )


def bool_values(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(False)

    return (
        normalize_protocol(series)
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
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


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return str(value)


def quantile_value(
    values: pd.Series,
    quantile: float,
) -> float | None:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return float(
        numeric.quantile(
            quantile
        )
    )


def read_split_molecule_ids(
    path: Path,
    current_spec: pd.DataFrame,
) -> set[int]:
    path = require_file(path)

    split = pd.read_csv(path)

    split = split.drop(
        columns=[
            column
            for column in split.columns
            if str(column).lower().startswith(
                "unnamed:"
            )
        ],
        errors="ignore",
    )

    if split.empty:
        return set()

    if "spec_id" in split.columns:
        id_column = "spec_id"
    else:
        id_column = split.columns[0]

    split_ids = set(
        normalize_id(
            split[id_column]
        )
        .dropna()
        .astype(str)
        .tolist()
    )

    mapping = current_spec[
        [
            "spec_id",
            "mol_id",
        ]
    ].copy()

    mapping["_spec_id_key"] = (
        normalize_id(
            mapping["spec_id"]
        )
    )

    matched = mapping[
        mapping[
            "_spec_id_key"
        ].isin(split_ids)
    ]

    if (
        matched[
            "_spec_id_key"
        ].nunique()
        != len(split_ids)
    ):
        raise RuntimeError(
            f"split ID映射不完整：{path}"
        )

    return set(
        matched["mol_id"]
        .astype(int)
        .tolist()
    )


def molecule_keys_for_ids(
    mol_ids: set[int],
    current_mol: pd.DataFrame,
) -> set[str]:
    subset = current_mol[
        current_mol["mol_id"]
        .astype(int)
        .isin(mol_ids)
    ]

    return set(
        connectivity_key(
            subset["inchikey_s"]
        )
        .dropna()
        .astype(str)
        .tolist()
    )


def annotation_entry_type_audit(
    series: pd.Series,
    row_limit: int = 3000,
) -> dict[str, Any]:
    outer_types = Counter(
        type(value).__name__
        for value in series
    )

    inner_types: Counter[str] = (
        Counter()
    )

    examples: list[str] = []

    inspected = 0

    for value in series:
        if inspected >= row_limit:
            break

        if not isinstance(
            value,
            (
                list,
                tuple,
                np.ndarray,
                pd.Series,
            ),
        ):
            continue

        inspected += 1

        for item in list(value)[:20]:
            inner_types[
                type(item).__name__
            ] += 1

            if len(examples) < 30:
                examples.append(
                    repr(item)[:300]
                )

    return {
        "outer_python_types": {
            str(key): int(value)
            for key, value
            in outer_types.items()
        },
        "inner_python_types": {
            str(key): int(value)
            for key, value
            in inner_types.items()
        },
        "examples": examples,
    }


for name, path in PATHS.items():
    require_file(path)

print()
print("=" * 100)
print("EXPERIMENT 5 SOURCE AUDIT")
print("=" * 100)

print("Loading current data...")
current_spec = pd.read_pickle(
    PATHS["current_spec"]
)

current_mol = pd.read_pickle(
    PATHS["current_mol"]
)

print("Loading complete NIST20 data...")
old_spec = pd.read_pickle(
    PATHS["old_spec"]
)

old_mol = pd.read_pickle(
    PATHS["old_mol"]
)

old_ann = pd.read_pickle(
    PATHS["old_ann"]
)

print(
    "current_spec:",
    current_spec.shape,
)

print(
    "current_mol:",
    current_mol.shape,
)

print(
    "old_spec:",
    old_spec.shape,
)

print(
    "old_mol:",
    old_mol.shape,
)

print(
    "old_ann:",
    old_ann.shape,
)


# =====================================================================
# Current split identities
# =====================================================================

random_train_mols = (
    read_split_molecule_ids(
        RANDOM_SPLIT_DIR
        / "train_ids.csv",
        current_spec,
    )
)

random_test_mols = (
    read_split_molecule_ids(
        RANDOM_SPLIT_DIR
        / "test_ids.csv",
        current_spec,
    )
)

scaffold_train_mols = (
    read_split_molecule_ids(
        SCAFFOLD_SPLIT_DIR
        / "train_ids.csv",
        current_spec,
    )
)

scaffold_test_mols = (
    read_split_molecule_ids(
        SCAFFOLD_SPLIT_DIR
        / "test_ids.csv",
        current_spec,
    )
)

current_all_keys = set(
    connectivity_key(
        current_mol["inchikey_s"]
    )
    .dropna()
    .astype(str)
    .tolist()
)

random_train_keys = (
    molecule_keys_for_ids(
        random_train_mols,
        current_mol,
    )
)

scaffold_train_keys = (
    molecule_keys_for_ids(
        scaffold_train_mols,
        current_mol,
    )
)

print()
print("Random train molecules:", len(random_train_mols))
print("Random test molecules:", len(random_test_mols))
print("Scaffold train molecules:", len(scaffold_train_mols))
print("Scaffold test molecules:", len(scaffold_test_mols))


# =====================================================================
# Experiment 5A: complete-NIST20 structure library audit
# =====================================================================

print()
print("=" * 100)
print("5A: BUILDING MODEL-COMPATIBLE STRUCTURE LIBRARY")
print("=" * 100)

fingerprint_generator = (
    rdFingerprintGenerator
    .GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
    )
)

current_element_set: set[str] = set()

for smiles in current_mol[
    "smiles"
].astype(str):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        continue

    current_element_set.update(
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    )

library_records = []
library_fingerprints: dict[
    str,
    Any,
] = {}

invalid_smiles_count = 0
missing_key_count = 0
unsupported_element_count = 0
non_single_count = 0
nonneutral_count = 0
radical_count = 0
invalid_mass_count = 0

for row in old_mol.itertuples(
    index=False
):
    smiles = getattr(
        row,
        "smiles",
        None,
    )

    key_value = getattr(
        row,
        "inchikey_s",
        None,
    )

    exact_mass = pd.to_numeric(
        pd.Series(
            [
                getattr(
                    row,
                    "exact_mw",
                    np.nan,
                )
            ]
        ),
        errors="coerce",
    ).iloc[0]

    single_value = getattr(
        row,
        "single_mol",
        False,
    )

    single_ok = bool_values(
        pd.Series([single_value])
    ).iloc[0]

    if not bool(single_ok):
        non_single_count += 1
        continue

    charge = pd.to_numeric(
        pd.Series(
            [
                getattr(
                    row,
                    "charge",
                    0,
                )
            ]
        ),
        errors="coerce",
    ).fillna(0).iloc[0]

    if float(charge) != 0.0:
        nonneutral_count += 1
        continue

    radicals = pd.to_numeric(
        pd.Series(
            [
                getattr(
                    row,
                    "num_radicals",
                    0,
                )
            ]
        ),
        errors="coerce",
    ).fillna(0).iloc[0]

    if float(radicals) != 0.0:
        radical_count += 1
        continue

    if (
        pd.isna(exact_mass)
        or not math.isfinite(
            float(exact_mass)
        )
        or float(exact_mass) <= 0.0
    ):
        invalid_mass_count += 1
        continue

    key = (
        str(key_value)
        .strip()
        .upper()
        .split("-")[0]
        if key_value is not None
        else ""
    )

    if (
        not key
        or key.lower() == "nan"
    ):
        missing_key_count += 1
        continue

    molecule = Chem.MolFromSmiles(
        str(smiles)
    )

    if molecule is None:
        invalid_smiles_count += 1
        continue

    elements = {
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    }

    if not elements.issubset(
        current_element_set
    ):
        unsupported_element_count += 1
        continue

    canonical_smiles = (
        Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    )

    fingerprint = (
        fingerprint_generator
        .GetFingerprint(
            molecule
        )
    )

    library_records.append(
        {
            "mol_id": int(
                getattr(row, "mol_id")
            ),
            "connectivity_key": key,
            "smiles": str(smiles),
            "canonical_smiles": (
                canonical_smiles
            ),
            "formula": getattr(
                row,
                "formula",
                None,
            ),
            "exact_mass": float(
                exact_mass
            ),
            "element_set": ".".join(
                sorted(elements)
            ),
        }
    )

    if key not in library_fingerprints:
        library_fingerprints[
            key
        ] = fingerprint

library_all = pd.DataFrame(
    library_records
)

if library_all.empty:
    raise RuntimeError(
        "没有获得可用候选结构。"
    )

duplicate_key_statistics = (
    library_all
    .groupby(
        "connectivity_key",
        as_index=False,
    )
    .agg(
        record_count=(
            "mol_id",
            "size",
        ),
        exact_mass_min=(
            "exact_mass",
            "min",
        ),
        exact_mass_max=(
            "exact_mass",
            "max",
        ),
    )
)

duplicate_key_statistics[
    "exact_mass_spread"
] = (
    duplicate_key_statistics[
        "exact_mass_max"
    ]
    - duplicate_key_statistics[
        "exact_mass_min"
    ]
)

library = (
    library_all
    .sort_values(
        [
            "connectivity_key",
            "mol_id",
        ]
    )
    .drop_duplicates(
        subset=[
            "connectivity_key"
        ],
        keep="first",
    )
    .sort_values(
        "exact_mass"
    )
    .reset_index(
        drop=True
    )
)

library_masses = (
    library["exact_mass"]
    .to_numpy(
        dtype=float
    )
)

library_keys = (
    library[
        "connectivity_key"
    ]
    .astype(str)
    .tolist()
)

library_fps = [
    library_fingerprints[key]
    for key in library_keys
]

library_key_set = set(
    library_keys
)

library_audit = {
    "raw_molecule_records": int(
        len(old_mol)
    ),
    "valid_records_before_connectivity_dedup": int(
        len(library_all)
    ),
    "unique_connectivity_structures": int(
        len(library)
    ),
    "current_supported_elements": sorted(
        current_element_set
    ),
    "filter_counts": {
        "invalid_smiles": int(
            invalid_smiles_count
        ),
        "missing_connectivity_key": int(
            missing_key_count
        ),
        "unsupported_elements": int(
            unsupported_element_count
        ),
        "not_single_component": int(
            non_single_count
        ),
        "nonneutral": int(
            nonneutral_count
        ),
        "radical": int(
            radical_count
        ),
        "invalid_exact_mass": int(
            invalid_mass_count
        ),
    },
    "duplicate_connectivity_key_count": int(
        (
            duplicate_key_statistics[
                "record_count"
            ] > 1
        ).sum()
    ),
    "maximum_duplicate_exact_mass_spread": float(
        duplicate_key_statistics[
            "exact_mass_spread"
        ].max()
    ),
    "protocol": {
        "mass_window_ppm": PPM,
        "negative_count_required": (
            NEGATIVE_COUNT_REQUIRED
        ),
        "fingerprint": (
            f"Morgan radius={MORGAN_RADIUS}, "
            f"bits={MORGAN_BITS}"
        ),
        "identity_deduplication": (
            "InChIKey first block"
        ),
    },
}

RETRIEVAL_LIBRARY_PATH.write_text(
    json.dumps(
        library_audit,
        indent=2,
        ensure_ascii=False,
        default=json_default,
    )
    + "\n",
    encoding="utf-8",
)

current_mol_lookup = (
    current_mol
    .set_index(
        "mol_id",
        drop=False,
    )
)

retrieval_rows = []

target_sets = {
    "random_test": (
        sorted(random_test_mols)
    ),
    "scaffold_test": (
        sorted(scaffold_test_mols)
    ),
}

for split_name, target_mol_ids in (
    target_sets.items()
):
    print()
    print(
        f"Auditing retrieval targets: "
        f"{split_name}"
    )

    for position, mol_id in enumerate(
        target_mol_ids,
        start=1,
    ):
        if mol_id not in (
            current_mol_lookup.index
        ):
            raise RuntimeError(
                f"缺少目标mol_id={mol_id}"
            )

        row = current_mol_lookup.loc[
            mol_id
        ]

        if isinstance(
            row,
            pd.DataFrame,
        ):
            row = row.iloc[0]

        target_key = (
            str(row["inchikey_s"])
            .strip()
            .upper()
            .split("-")[0]
        )

        target_mass = float(
            row["exact_mw"]
        )

        target_molecule = (
            Chem.MolFromSmiles(
                str(row["smiles"])
            )
        )

        target_valid = (
            target_molecule is not None
            and math.isfinite(
                target_mass
            )
        )

        lower_mass = (
            target_mass
            * (
                1.0
                - PPM * 1.0e-6
            )
        )

        upper_mass = (
            target_mass
            * (
                1.0
                + PPM * 1.0e-6
            )
        )

        left = int(
            np.searchsorted(
                library_masses,
                lower_mass,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                library_masses,
                upper_mass,
                side="right",
            )
        )

        candidate_indices = [
            index
            for index in range(
                left,
                right,
            )
            if library_keys[index]
            != target_key
        ]

        negative_count = int(
            len(candidate_indices)
        )

        pool_ready = bool(
            target_valid
            and target_key
            in library_key_set
            and negative_count
            >= NEGATIVE_COUNT_REQUIRED
        )

        top1_similarity = np.nan
        top49_mean_similarity = np.nan
        top49_median_similarity = np.nan
        top49_min_similarity = np.nan

        if (
            target_valid
            and negative_count > 0
        ):
            target_fp = (
                fingerprint_generator
                .GetFingerprint(
                    target_molecule
                )
            )

            candidate_fps = [
                library_fps[index]
                for index
                in candidate_indices
            ]

            similarities = np.asarray(
                DataStructs
                .BulkTanimotoSimilarity(
                    target_fp,
                    candidate_fps,
                ),
                dtype=float,
            )

            similarities = np.sort(
                similarities
            )[::-1]

            top1_similarity = float(
                similarities[0]
            )

            top_values = similarities[
                :min(
                    NEGATIVE_COUNT_REQUIRED,
                    len(similarities),
                )
            ]

            top49_mean_similarity = (
                float(top_values.mean())
            )

            top49_median_similarity = (
                float(
                    np.median(top_values)
                )
            )

            top49_min_similarity = (
                float(top_values.min())
            )

        retrieval_rows.append(
            {
                "split": split_name,
                "target_mol_id": int(
                    mol_id
                ),
                "target_connectivity_key": (
                    target_key
                ),
                "target_exact_mass": (
                    target_mass
                ),
                "target_structure_valid": bool(
                    target_valid
                ),
                "target_present_in_library": bool(
                    target_key
                    in library_key_set
                ),
                "mass_window_unique_negative_count": (
                    negative_count
                ),
                "candidate_pool_ready_1_plus_49": (
                    pool_ready
                ),
                "top1_negative_tanimoto": (
                    top1_similarity
                ),
                "top49_mean_tanimoto": (
                    top49_mean_similarity
                ),
                "top49_median_tanimoto": (
                    top49_median_similarity
                ),
                "top49_min_tanimoto": (
                    top49_min_similarity
                ),
            }
        )

        if (
            position % 100 == 0
            or position
            == len(target_mol_ids)
        ):
            print(
                f"{split_name}: "
                f"{position}/"
                f"{len(target_mol_ids)}",
                flush=True,
            )

retrieval_detail = pd.DataFrame(
    retrieval_rows
)

retrieval_detail.to_csv(
    RETRIEVAL_DETAIL_PATH,
    index=False,
)

retrieval_summary_rows = []

for split_name in (
    "random_test",
    "scaffold_test",
):
    subset = retrieval_detail[
        retrieval_detail["split"]
        == split_name
    ]

    ready_count = int(
        subset[
            "candidate_pool_ready_1_plus_49"
        ].sum()
    )

    retrieval_summary_rows.append(
        {
            "split": split_name,
            "target_molecule_count": int(
                len(subset)
            ),
            "target_present_count": int(
                subset[
                    "target_present_in_library"
                ].sum()
            ),
            "pool_ready_count": (
                ready_count
            ),
            "pool_ready_percent": float(
                100.0
                * ready_count
                / len(subset)
            ),
            "negative_count_min": (
                quantile_value(
                    subset[
                        "mass_window_unique_negative_count"
                    ],
                    0.0,
                )
            ),
            "negative_count_q25": (
                quantile_value(
                    subset[
                        "mass_window_unique_negative_count"
                    ],
                    0.25,
                )
            ),
            "negative_count_median": (
                quantile_value(
                    subset[
                        "mass_window_unique_negative_count"
                    ],
                    0.5,
                )
            ),
            "negative_count_q75": (
                quantile_value(
                    subset[
                        "mass_window_unique_negative_count"
                    ],
                    0.75,
                )
            ),
            "negative_count_max": (
                quantile_value(
                    subset[
                        "mass_window_unique_negative_count"
                    ],
                    1.0,
                )
            ),
            "ready_top1_tanimoto_mean": float(
                subset.loc[
                    subset[
                        "candidate_pool_ready_1_plus_49"
                    ],
                    "top1_negative_tanimoto",
                ].mean()
            )
            if ready_count
            else np.nan,
            "ready_top49_min_tanimoto_mean": float(
                subset.loc[
                    subset[
                        "candidate_pool_ready_1_plus_49"
                    ],
                    "top49_min_tanimoto",
                ].mean()
            )
            if ready_count
            else np.nan,
        }
    )

retrieval_summary = pd.DataFrame(
    retrieval_summary_rows
)

retrieval_summary.to_csv(
    RETRIEVAL_SUMMARY_PATH,
    index=False,
)


# =====================================================================
# Experiment 5B: independent native-annotation benchmark audit
# =====================================================================

print()
print("=" * 100)
print("5B: NATIVE-ANNOTATION BENCHMARK AUDIT")
print("=" * 100)

old_spec_meta_columns = [
    "spec_id",
    "mol_id",
    "dset",
    "dset_spec_id",
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "col_gas",
    "res",
    "ace",
    "nce",
    "prec_mz",
]

old_spec_meta = old_spec[
    old_spec_meta_columns
].copy()

old_spec_meta[
    "_dset_key"
] = normalize_protocol(
    old_spec_meta["dset"]
)

old_spec_meta[
    "_dset_spec_id_key"
] = normalize_id(
    old_spec_meta[
        "dset_spec_id"
    ]
)

old_spec_meta[
    "_prec_type_key"
] = normalize_protocol(
    old_spec_meta[
        "prec_type"
    ]
)

ann = old_ann.copy()

ann[
    "_dset_key"
] = normalize_protocol(
    ann["dset"]
)

ann[
    "_dset_spec_id_key"
] = normalize_id(
    ann["dset_spec_id"]
)

ann[
    "_prec_type_key"
] = normalize_protocol(
    ann["prec_type"]
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

annotation_columns = [
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

annotated = old_spec_meta.merge(
    ann[
        annotation_columns
    ],
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
        "old_ann与old_spec没有完全对齐："
        f"{len(annotated)} vs "
        f"{len(old_ann)}"
    )

mol_id_mismatch_count = int(
    (
        annotated["mol_id"]
        .astype(int)
        != annotated[
            "annotation_source_mol_id"
        ].astype(int)
    ).sum()
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

if old_mol_meta[
    "mol_id"
].duplicated().any():
    raise RuntimeError(
        "old_mol中的mol_id不唯一。"
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

protocol_columns = [
    "dset",
    "prec_type",
    "inst_type",
    "frag_mode",
    "spec_type",
    "ion_mode",
    "res",
]

protocol_allowed = {}
protocol_mask = pd.Series(
    True,
    index=annotated.index,
)

for column in protocol_columns:
    allowed = set(
        normalize_protocol(
            current_spec[column]
        )
        .dropna()
        .astype(str)
        .tolist()
    )

    protocol_allowed[
        column
    ] = sorted(allowed)

    protocol_mask &= (
        normalize_protocol(
            annotated[column]
        )
        .isin(allowed)
    )

selection_flow = []

def record_selection(
    name: str,
    frame: pd.DataFrame,
) -> None:
    selection_flow.append(
        {
            "stage": name,
            "spectrum_count": int(
                len(frame)
            ),
            "molecule_count": int(
                frame[
                    "connectivity_key"
                ].nunique()
            )
            if (
                "connectivity_key"
                in frame.columns
            )
            else int(
                frame[
                    "mol_id"
                ].nunique()
            ),
        }
    )


record_selection(
    "all_native_annotations",
    annotated,
)

eligible = annotated[
    protocol_mask
].copy()

record_selection(
    "protocol_compatible",
    eligible,
)

eligible["ace"] = pd.to_numeric(
    eligible["ace"],
    errors="coerce",
)

eligible = eligible[
    eligible["ace"].notna()
    & np.isfinite(
        eligible["ace"]
    )
].copy()

record_selection(
    "ace_available",
    eligible,
)

eligible = eligible[
    eligible[
        "connectivity_key"
    ].isin(
        library_key_set
    )
].copy()

record_selection(
    "model_compatible_structure",
    eligible,
)

annotation_fields = [
    "ann_peak_mzs",
    "ann_products",
    "ann_losses",
    "ann_isotopes",
    "ann_exact_mzs",
]

for field in annotation_fields:
    eligible[
        f"{field}_length"
    ] = eligible[field].map(
        sequence_length
    )

eligible = eligible[
    eligible[
        "ann_peak_mzs_length"
    ].gt(0)
    & eligible[
        "ann_products_length"
    ].gt(0)
    & eligible[
        "ann_exact_mzs_length"
    ].gt(0)
].copy()

record_selection(
    "nonempty_peak_product_exact_annotations",
    eligible,
)

eligible[
    "core_annotation_lengths_equal"
] = (
    (
        eligible[
            "ann_peak_mzs_length"
        ]
        == eligible[
            "ann_products_length"
        ]
    )
    & (
        eligible[
            "ann_peak_mzs_length"
        ]
        == eligible[
            "ann_exact_mzs_length"
        ]
    )
)

eligible[
    "all_five_annotation_lengths_equal"
] = (
    eligible[
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

current_dset_spec_ids = set(
    normalize_id(
        current_spec[
            "dset_spec_id"
        ]
    )
    .dropna()
    .astype(str)
    .tolist()
)

eligible[
    "is_current_safe_spectrum"
] = (
    eligible[
        "_dset_spec_id_key"
    ].isin(
        current_dset_spec_ids
    )
)

subsets = {
    "protocol_compatible_annotated": (
        eligible
    ),
    "random_training_connectivity_disjoint": (
        eligible[
            ~eligible[
                "connectivity_key"
            ].isin(
                random_train_keys
            )
        ]
    ),
    "scaffold_training_connectivity_disjoint": (
        eligible[
            ~eligible[
                "connectivity_key"
            ].isin(
                scaffold_train_keys
            )
        ]
    ),
    "outside_all_current_2274_connectivities": (
        eligible[
            ~eligible[
                "connectivity_key"
            ].isin(
                current_all_keys
            )
        ]
    ),
}

subsets[
    "outside_all_current_and_core_lengths_equal"
] = (
    subsets[
        "outside_all_current_2274_connectivities"
    ][
        subsets[
            "outside_all_current_2274_connectivities"
        ][
            "core_annotation_lengths_equal"
        ]
    ].copy()
)


def summarize_formula_subset(
    name: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "subset": name,
            "spectrum_count": 0,
            "connectivity_count": 0,
            "ace_min": np.nan,
            "ace_median": np.nan,
            "ace_max": np.nan,
            "spectra_per_connectivity_median": (
                np.nan
            ),
            "annotated_peak_entries_total": 0,
            "annotated_peak_entries_median": (
                np.nan
            ),
            "core_length_equal_percent": (
                np.nan
            ),
            "current_safe_spectrum_overlap": 0,
        }

    spectra_per_structure = (
        frame
        .groupby(
            "connectivity_key"
        )
        .size()
    )

    return {
        "subset": name,
        "spectrum_count": int(
            len(frame)
        ),
        "connectivity_count": int(
            frame[
                "connectivity_key"
            ].nunique()
        ),
        "ace_min": float(
            frame["ace"].min()
        ),
        "ace_median": float(
            frame["ace"].median()
        ),
        "ace_max": float(
            frame["ace"].max()
        ),
        "spectra_per_connectivity_median": float(
            spectra_per_structure.median()
        ),
        "annotated_peak_entries_total": int(
            frame[
                "ann_peak_mzs_length"
            ].sum()
        ),
        "annotated_peak_entries_median": float(
            frame[
                "ann_peak_mzs_length"
            ].median()
        ),
        "core_length_equal_percent": float(
            100.0
            * frame[
                "core_annotation_lengths_equal"
            ].mean()
        ),
        "current_safe_spectrum_overlap": int(
            frame[
                "is_current_safe_spectrum"
            ].sum()
        ),
    }


formula_summary = pd.DataFrame(
    [
        summarize_formula_subset(
            name,
            frame,
        )
        for name, frame
        in subsets.items()
    ]
)

formula_summary.to_csv(
    FORMULA_SUMMARY_PATH,
    index=False,
)

formula_manifest = subsets[
    "outside_all_current_and_core_lengths_equal"
].copy()

if (
    formula_manifest[
        "connectivity_key"
    ].isin(
        current_all_keys
    ).any()
):
    raise RuntimeError(
        "5B候选benchmark仍存在当前分子重叠。"
    )

formula_manifest.to_pickle(
    FORMULA_MANIFEST_PATH
)

manifest_csv_columns = [
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
    "res",
    "ace",
    "nce",
    "prec_mz",
    "ann_peak_mzs_length",
    "ann_products_length",
    "ann_losses_length",
    "ann_isotopes_length",
    "ann_exact_mzs_length",
]

formula_manifest[
    manifest_csv_columns
].to_csv(
    FORMULA_MANIFEST_CSV_PATH,
    index=False,
)

formula_schema = {
    "join": {
        "native_annotation_row_count": int(
            len(old_ann)
        ),
        "joined_row_count": int(
            len(annotated)
        ),
        "mol_id_mismatch_count": (
            mol_id_mismatch_count
        ),
    },
    "protocol_allowed_values": (
        protocol_allowed
    ),
    "selection_flow": (
        selection_flow
    ),
    "annotation_field_types": {
        field: (
            annotation_entry_type_audit(
                eligible[field]
            )
        )
        for field in annotation_fields
    },
    "length_consistency": {
        "eligible_row_count": int(
            len(eligible)
        ),
        "core_peak_product_exact_equal_count": int(
            eligible[
                "core_annotation_lengths_equal"
            ].sum()
        ),
        "core_peak_product_exact_equal_percent": float(
            100.0
            * eligible[
                "core_annotation_lengths_equal"
            ].mean()
        )
        if len(eligible)
        else None,
        "all_five_equal_count": int(
            eligible[
                "all_five_annotation_lengths_equal"
            ].sum()
        ),
        "all_five_equal_percent": float(
            100.0
            * eligible[
                "all_five_annotation_lengths_equal"
            ].mean()
        )
        if len(eligible)
        else None,
    },
    "leakage_audit": {
        "current_connectivity_count": int(
            len(current_all_keys)
        ),
        "official_manifest_current_connectivity_overlap": int(
            len(
                set(
                    formula_manifest[
                        "connectivity_key"
                    ]
                    .astype(str)
                )
                & current_all_keys
            )
        ),
        "official_manifest_current_spectrum_overlap": int(
            formula_manifest[
                "is_current_safe_spectrum"
            ].sum()
        ),
    },
    "outputs": {
        "manifest_pkl": str(
            FORMULA_MANIFEST_PATH.resolve()
        ),
        "manifest_csv": str(
            FORMULA_MANIFEST_CSV_PATH.resolve()
        ),
    },
}

FORMULA_SCHEMA_PATH.write_text(
    json.dumps(
        formula_schema,
        indent=2,
        ensure_ascii=False,
        default=json_default,
    )
    + "\n",
    encoding="utf-8",
)

final_report = {
    "status": "audit_complete",
    "experiment_5a": {
        "library": library_audit,
        "summary": (
            retrieval_summary
            .to_dict(
                orient="records"
            )
        ),
    },
    "experiment_5b": {
        "summary": (
            formula_summary
            .to_dict(
                orient="records"
            )
        ),
        "official_candidate_manifest_spectra": int(
            len(formula_manifest)
        ),
        "official_candidate_manifest_connectivities": int(
            formula_manifest[
                "connectivity_key"
            ].nunique()
        ),
        "current_connectivity_overlap": int(
            formula_schema[
                "leakage_audit"
            ][
                "official_manifest_current_connectivity_overlap"
            ]
        ),
    },
    "outputs": {
        "retrieval_detail": str(
            RETRIEVAL_DETAIL_PATH.resolve()
        ),
        "retrieval_summary": str(
            RETRIEVAL_SUMMARY_PATH.resolve()
        ),
        "formula_summary": str(
            FORMULA_SUMMARY_PATH.resolve()
        ),
        "formula_schema": str(
            FORMULA_SCHEMA_PATH.resolve()
        ),
        "formula_manifest": str(
            FORMULA_MANIFEST_PATH.resolve()
        ),
    },
}

FINAL_REPORT_PATH.write_text(
    json.dumps(
        final_report,
        indent=2,
        ensure_ascii=False,
        default=json_default,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("5A RETRIEVAL COVERAGE SUMMARY")
print("=" * 100)
print(
    retrieval_summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("5B FORMULA BENCHMARK SUMMARY")
print("=" * 100)
print(
    formula_summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)
print(
    "Retrieval detail:",
    RETRIEVAL_DETAIL_PATH.resolve(),
)
print(
    "Retrieval summary:",
    RETRIEVAL_SUMMARY_PATH.resolve(),
)
print(
    "Formula summary:",
    FORMULA_SUMMARY_PATH.resolve(),
)
print(
    "Formula schema:",
    FORMULA_SCHEMA_PATH.resolve(),
)
print(
    "Formula manifest:",
    FORMULA_MANIFEST_PATH.resolve(),
)
print(
    "Final report:",
    FINAL_REPORT_PATH.resolve(),
)

print()
print("EXPERIMENT_5_SOURCE_AUDIT_COMPLETE")
