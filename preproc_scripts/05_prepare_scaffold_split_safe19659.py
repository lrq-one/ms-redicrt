#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SPLIT = (
    ROOT
    / "data/split/"
    "nist20_qtof_cid_safe19659_qcv1_trainonly"
)

OUTPUT_SPLIT = (
    ROOT
    / "data/split/"
    "nist20_qtof_cid_safe19659_"
    "scaffold60_20_20_seed42"
)

MOL_PATH = (
    ROOT
    / "data/proc/"
    "nist20_qtof_cid_safe19659/"
    "mol_df.pkl"
)

SEED = 42

RATIOS = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20,
}


def clean_scaffold(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return None

    return text


def load_pool():
    frames = []

    for split in (
        "train",
        "val",
        "test",
    ):
        path = (
            SOURCE_SPLIT
            / f"{split}_ids.csv"
        )

        frames.append(
            pd.read_csv(path)[
                [
                    "spec_id",
                    "mol_id",
                    "group_id",
                ]
            ]
        )

    pool = pd.concat(
        frames,
        ignore_index=True,
    )

    if pool["spec_id"].duplicated().any():
        raise RuntimeError(
            "源split存在重复spec_id"
        )

    return pool


def attach_scaffolds(pool):
    mol_df = pd.read_pickle(
        MOL_PATH
    )

    if "mol_id" not in mol_df.columns:
        if mol_df.index.name == "mol_id":
            mol_df = mol_df.reset_index()
        else:
            raise RuntimeError(
                "mol_df中没有mol_id"
            )

    if "scaffold" not in mol_df.columns:
        raise RuntimeError(
            "mol_df中没有scaffold"
        )

    mapping = (
        mol_df[
            [
                "mol_id",
                "scaffold",
            ]
        ]
        .drop_duplicates(
            subset=["mol_id"]
        )
        .copy()
    )

    mapping["murcko_scaffold"] = (
        mapping["scaffold"]
        .map(clean_scaffold)
    )

    mapping["is_acyclic"] = (
        mapping["murcko_scaffold"]
        .isna()
    )

    mapping["scaffold_group"] = (
        mapping.apply(
            lambda row: (
                "__ACYCLIC_SINGLETON__"
                + str(row["mol_id"])
                if row["is_acyclic"]
                else row[
                    "murcko_scaffold"
                ]
            ),
            axis=1,
        )
    )

    result = pool.merge(
        mapping[
            [
                "mol_id",
                "murcko_scaffold",
                "is_acyclic",
                "scaffold_group",
            ]
        ],
        on="mol_id",
        how="left",
        validate="many_to_one",
    )

    if result[
        "scaffold_group"
    ].isna().any():
        raise RuntimeError(
            "部分分子没有scaffold映射"
        )

    return result


def assign_groups(pool):
    groups = (
        pool
        .groupby(
            "scaffold_group",
            as_index=False,
        )
        .agg(
            spectrum_count=(
                "spec_id",
                "nunique",
            ),
            molecule_count=(
                "mol_id",
                "nunique",
            ),
            is_acyclic=(
                "is_acyclic",
                "max",
            ),
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    groups["tie"] = rng.random(
        len(groups)
    )

    groups = (
        groups
        .sort_values(
            [
                "spectrum_count",
                "molecule_count",
                "tie",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    total_spectra = pool[
        "spec_id"
    ].nunique()

    total_molecules = pool[
        "mol_id"
    ].nunique()

    target_spectra = {
        split:
            total_spectra * ratio
        for split, ratio
        in RATIOS.items()
    }

    target_molecules = {
        split:
            total_molecules * ratio
        for split, ratio
        in RATIOS.items()
    }

    current_spectra = {
        split: 0
        for split in RATIOS
    }

    current_molecules = {
        split: 0
        for split in RATIOS
    }

    split_order = [
        "train",
        "val",
        "test",
    ]

    assignments = []

    for row in groups.itertuples(
        index=False
    ):
        choices = []

        for priority, split in enumerate(
            split_order
        ):
            spec_fill = (
                current_spectra[split]
                + row.spectrum_count
            ) / target_spectra[split]

            mol_fill = (
                current_molecules[split]
                + row.molecule_count
            ) / target_molecules[split]

            overflow = (
                max(
                    0.0,
                    spec_fill - 1.0,
                )
                + max(
                    0.0,
                    mol_fill - 1.0,
                )
            )

            score = (
                0.65 * spec_fill
                + 0.35 * mol_fill
                + 2.0 * overflow
            )

            choices.append(
                (
                    score,
                    priority,
                    split,
                )
            )

        _, _, selected = min(
            choices
        )

        current_spectra[
            selected
        ] += int(
            row.spectrum_count
        )

        current_molecules[
            selected
        ] += int(
            row.molecule_count
        )

        assignments.append(
            {
                "scaffold_group":
                    row.scaffold_group,
                "split":
                    selected,
            }
        )

    return pd.DataFrame(
        assignments
    )


def count_overlap(
    frame,
    column,
    left,
    right,
):
    left_values = set(
        frame.loc[
            frame["split"] == left,
            column,
        ].dropna().tolist()
    )

    right_values = set(
        frame.loc[
            frame["split"] == right,
            column,
        ].dropna().tolist()
    )

    return len(
        left_values
        & right_values
    )


def main():
    pool = attach_scaffolds(
        load_pool()
    )

    assignments = assign_groups(
        pool
    )

    pool = pool.merge(
        assignments,
        on="scaffold_group",
        how="left",
        validate="many_to_one",
    )

    OUTPUT_SPLIT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        frame = pool[
            pool["split"] == split
        ]

        frame[
            [
                "spec_id",
                "mol_id",
                "group_id",
            ]
        ].sort_values(
            "spec_id"
        ).to_csv(
            OUTPUT_SPLIT
            / f"{split}_ids.csv",
            index=False,
        )

        nonempty = frame[
            ~frame["is_acyclic"]
        ]

        summary[split] = {
            "spectra":
                int(
                    frame[
                        "spec_id"
                    ].nunique()
                ),
            "molecules":
                int(
                    frame[
                        "mol_id"
                    ].nunique()
                ),
            "scaffold_groups":
                int(
                    frame[
                        "scaffold_group"
                    ].nunique()
                ),
            "nonempty_murcko_scaffolds":
                int(
                    nonempty[
                        "murcko_scaffold"
                    ].nunique()
                ),
            "spectrum_fraction":
                float(
                    frame[
                        "spec_id"
                    ].nunique()
                    / pool[
                        "spec_id"
                    ].nunique()
                ),
        }

    pd.DataFrame(
        columns=[
            "spec_id",
            "mol_id",
            "group_id",
        ]
    ).to_csv(
        OUTPUT_SPLIT
        / "secondary_ids.csv",
        index=False,
    )

    pool.to_csv(
        OUTPUT_SPLIT
        / "all_ids_with_scaffold.csv",
        index=False,
    )

    assignments.to_csv(
        OUTPUT_SPLIT
        / "scaffold_assignments.csv",
        index=False,
    )

    overlaps = {}

    for left, right in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        for column in (
            "spec_id",
            "mol_id",
            "scaffold_group",
            "murcko_scaffold",
        ):
            overlaps[
                f"{left}_{right}_{column}"
            ] = count_overlap(
                pool,
                column,
                left,
                right,
            )

    if any(
        value != 0
        for value in overlaps.values()
    ):
        raise RuntimeError(
            f"存在跨split重叠：{overlaps}"
        )

    for split in (
        "train",
        "val",
        "test",
    ):
        fraction = summary[
            split
        ]["spectrum_fraction"]

        if abs(
            fraction - RATIOS[split]
        ) > 0.03:
            raise RuntimeError(
                f"{split}比例偏差过大："
                f"{fraction}"
            )

        if summary[
            split
        ][
            "nonempty_murcko_scaffolds"
        ] < 10:
            raise RuntimeError(
                f"{split}非空骨架数量过少"
            )

    audit = {
        "split_type":
            (
                "bemis_murcko_scaffold_"
                "disjoint_with_acyclic_"
                "singletons"
            ),
        "split_seed":
            SEED,
        "target_ratios":
            RATIOS,
        "source_population": {
            "spectra":
                int(
                    pool[
                        "spec_id"
                    ].nunique()
                ),
            "molecules":
                int(
                    pool[
                        "mol_id"
                    ].nunique()
                ),
            "acyclic_molecules":
                int(
                    pool.loc[
                        pool[
                            "is_acyclic"
                        ],
                        "mol_id",
                    ].nunique()
                ),
        },
        "splits":
            summary,
        "overlap":
            overlaps,
        "all_disjoint":
            True,
        "source_population_preserved":
            (
                pool[
                    "spec_id"
                ].nunique()
                == 19659
            ),
        "acyclic_policy":
            "molecule-specific singleton",
        "test_used_for_split_selection":
            False,
    }

    (
        OUTPUT_SPLIT
        / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("BEMIS-MURCKO SCAFFOLD SPLIT")
    print("=" * 100)

    for split in (
        "train",
        "val",
        "test",
    ):
        info = summary[split]

        print(
            f"{split:<5} "
            f"spectra={info['spectra']:<6} "
            f"molecules={info['molecules']:<5} "
            f"groups={info['scaffold_groups']:<5} "
            f"murcko={info['nonempty_murcko_scaffolds']:<5} "
            f"fraction={info['spectrum_fraction']:.4f}"
        )

    print("overlap =", overlaps)
    print("SCAFFOLD_SPLIT_OK")
    print("=" * 100)


if __name__ == "__main__":
    main()
