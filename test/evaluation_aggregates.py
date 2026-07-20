from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def _split_mapping_path(
    config: dict,
    split: str,
) -> Path:
    split_dp = Path(
        str(config["split_dp"])
    )

    if not split_dp.is_absolute():
        split_dp = ROOT / split_dp

    path = split_dp / f"{split}_ids.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split mapping: {path}"
        )

    return path


def _attach_molecule_ids(
    detail: pd.DataFrame,
    config: dict,
    split: str,
) -> pd.DataFrame:
    required_detail = {
        "spec_id",
        "ce",
        "cos",
        "jss",
    }

    missing_detail = (
        required_detail
        - set(detail.columns)
    )

    if missing_detail:
        raise RuntimeError(
            "Per-spectrum结果缺少字段："
            + ", ".join(
                sorted(missing_detail)
            )
        )

    mapping_path = _split_mapping_path(
        config,
        split,
    )

    mapping = pd.read_csv(
        mapping_path
    )

    required_mapping = {
        "spec_id",
        "mol_id",
    }

    missing_mapping = (
        required_mapping
        - set(mapping.columns)
    )

    if missing_mapping:
        raise RuntimeError(
            "Split映射缺少字段："
            + ", ".join(
                sorted(missing_mapping)
            )
        )

    detail = detail.copy()
    mapping = mapping[
        ["spec_id", "mol_id"]
    ].copy()

    detail["spec_id"] = pd.to_numeric(
        detail["spec_id"],
        errors="raise",
    ).astype(np.int64)

    mapping["spec_id"] = pd.to_numeric(
        mapping["spec_id"],
        errors="raise",
    ).astype(np.int64)

    conflicts = (
        mapping
        .groupby("spec_id")["mol_id"]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    if len(conflicts) > 0:
        raise RuntimeError(
            "同一个spec_id对应多个mol_id："
            + repr(
                conflicts.index[
                    :20
                ].tolist()
            )
        )

    mapping = mapping.drop_duplicates(
        subset=["spec_id"]
    )

    merged = detail.merge(
        mapping,
        on="spec_id",
        how="left",
        validate="many_to_one",
    )

    missing_molecule = merged[
        "mol_id"
    ].isna()

    if missing_molecule.any():
        missing_ids = (
            merged.loc[
                missing_molecule,
                "spec_id",
            ]
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            "部分谱图没有mol_id映射："
            + repr(missing_ids)
        )

    merged[
        "spectra_per_molecule"
    ] = (
        merged
        .groupby("mol_id")["spec_id"]
        .transform("size")
        .astype(int)
    )

    return merged


def _macro_summary(
    per_molecule: pd.DataFrame,
    spectrum_count: int,
) -> dict:
    return {
        "molecule_count":
            int(len(per_molecule)),

        "spectrum_count":
            int(spectrum_count),

        "cosine":
            float(
                per_molecule[
                    "cosine"
                ].mean()
            ),

        "jss":
            float(
                per_molecule[
                    "jss"
                ].mean()
            ),

        "mean_ce":
            float(
                per_molecule[
                    "mean_ce"
                ].mean()
            ),

        "mean_spectra_per_molecule":
            float(
                per_molecule[
                    "spectrum_count"
                ].mean()
            ),

        "median_spectra_per_molecule":
            float(
                per_molecule[
                    "spectrum_count"
                ].median()
            ),
    }


def _highest_ce_summary(
    frame: pd.DataFrame,
) -> dict:
    if len(frame) == 0:
        return {
            "molecule_count": 0,
            "spectrum_count": 0,
            "cosine": None,
            "jss": None,
            "mean_max_ce": None,
        }

    return {
        "molecule_count":
            int(
                frame[
                    "mol_id"
                ].nunique()
            ),

        "spectrum_count":
            int(
                frame[
                    "tie_spectrum_count"
                ].sum()
            ),

        "cosine":
            float(
                frame[
                    "cosine"
                ].mean()
            ),

        "jss":
            float(
                frame[
                    "jss"
                ].mean()
            ),

        "mean_max_ce":
            float(
                frame[
                    "max_ce"
                ].mean()
            ),
    }


def evaluate_molecule_aggregates(
    detail: pd.DataFrame,
    config: dict,
    split: str,
    prefix: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged = _attach_molecule_ids(
        detail=detail,
        config=config,
        split=split,
    )

    per_molecule = (
        merged
        .groupby(
            "mol_id",
            as_index=False,
        )
        .agg(
            spectrum_count=(
                "spec_id",
                "size",
            ),
            mean_ce=(
                "ce",
                "mean",
            ),
            min_ce=(
                "ce",
                "min",
            ),
            max_ce=(
                "ce",
                "max",
            ),
            cosine=(
                "cos",
                "mean",
            ),
            jss=(
                "jss",
                "mean",
            ),
        )
    )

    molecule_macro = _macro_summary(
        per_molecule=per_molecule,
        spectrum_count=len(merged),
    )

    per_molecule[
        "spectrum_count_group"
    ] = pd.cut(
        per_molecule[
            "spectrum_count"
        ],
        bins=[
            0,
            1,
            3,
            7,
            np.inf,
        ],
        labels=[
            "1",
            "2-3",
            "4-7",
            "8+",
        ],
        include_lowest=True,
        right=True,
    )

    count_groups = (
        per_molecule
        .groupby(
            "spectrum_count_group",
            observed=True,
        )
        .agg(
            molecule_count=(
                "mol_id",
                "nunique",
            ),
            spectrum_count=(
                "spectrum_count",
                "sum",
            ),
            mean_spectra_per_molecule=(
                "spectrum_count",
                "mean",
            ),
            cosine=(
                "cosine",
                "mean",
            ),
            jss=(
                "jss",
                "mean",
            ),
            mean_ce=(
                "mean_ce",
                "mean",
            ),
        )
        .reset_index()
    )

    max_ce = (
        merged
        .groupby("mol_id")["ce"]
        .transform("max")
    )

    highest_rows = merged[
        np.isclose(
            merged["ce"],
            max_ce,
            rtol=0.0,
            atol=1.0e-8,
        )
    ].copy()

    highest_per_molecule = (
        highest_rows
        .groupby(
            "mol_id",
            as_index=False,
        )
        .agg(
            tie_spectrum_count=(
                "spec_id",
                "size",
            ),
            max_ce=(
                "ce",
                "max",
            ),
            cosine=(
                "cos",
                "mean",
            ),
            jss=(
                "jss",
                "mean",
            ),
        )
    )

    highest_ce_all = (
        _highest_ce_summary(
            highest_per_molecule
        )
    )

    highest_ce_gt40 = (
        _highest_ce_summary(
            highest_per_molecule[
                highest_per_molecule[
                    "max_ce"
                ] > 40.0
            ]
        )
    )

    merged.to_csv(
        output_dir
        / (
            f"{prefix}_per_spectrum_"
            "with_molecule.csv"
        ),
        index=False,
    )

    per_molecule.to_csv(
        output_dir
        / f"{prefix}_per_molecule.csv",
        index=False,
    )

    count_groups.to_csv(
        output_dir
        / (
            f"{prefix}_molecule_"
            "spectrum_count_groups.csv"
        ),
        index=False,
    )

    highest_per_molecule.to_csv(
        output_dir
        / (
            f"{prefix}_highest_ce_"
            "per_molecule.csv"
        ),
        index=False,
    )

    result = {
        "definitions": {
            "spectrum_micro":
                (
                    "Arithmetic mean over "
                    "individual spectra; "
                    "each spectrum has equal weight."
                ),

            "molecule_macro":
                (
                    "First average spectra "
                    "within each molecule, "
                    "then average molecules; "
                    "each molecule has equal weight."
                ),

            "highest_ce_per_molecule":
                (
                    "For each molecule, use "
                    "the spectrum at its maximum "
                    "observed collision energy. "
                    "Tied maximum-CE spectra are "
                    "averaged first."
                ),
        },

        "molecule_macro":
            molecule_macro,

        "spectrum_count_groups":
            count_groups.to_dict(
                orient="records"
            ),

        "highest_ce_per_molecule":
            highest_ce_all,

        "highest_ce_gt40":
            highest_ce_gt40,
    }

    result = _json_safe(result)

    result_path = (
        output_dir
        / (
            f"{prefix}_molecule_"
            "aggregates.json"
        )
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

    return result
