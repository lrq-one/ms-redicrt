from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SEEDS = (42, 43, 44)
SPLITS = ("validation", "test")

REQUIRED_COLUMNS = {
    "ce_bucket",
    "spec_count",
    "mean_ce",
    "cos_0.01",
    "jss_0.01",
    "chun_10ppm",
}

SUMMARY_METRICS = (
    "cos_0.01",
    "jss_0.01",
    "chun_10ppm",
    "chun_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "汇总三个随机种子的ACE分层CBIN、JSS和CHUN结果。"
        )
    )
    parser.add_argument(
        "--experiment-root",
        required=True,
        type=Path,
        help=(
            "例如 runs/experiments/"
            "molecule_disjoint_3seeds"
        ),
    )
    return parser.parse_args()


def to_json_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    return json.loads(
        dataframe.to_json(
            orient="records",
            double_precision=15,
        )
    )


def main() -> None:
    args = parse_args()

    experiment_root = (
        args.experiment_root.expanduser().resolve()
    )

    if not experiment_root.is_dir():
        raise FileNotFoundError(
            f"实验目录不存在：{experiment_root}"
        )

    root_output_dir = (
        experiment_root / "ace_stratified"
    )
    root_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: list[pd.DataFrame] = []

    for seed in SEEDS:
        seed_dir = (
            experiment_root / f"seed_{seed}"
        )

        if not seed_dir.is_dir():
            raise FileNotFoundError(
                f"缺少seed目录：{seed_dir}"
            )

        seed_output_dir = (
            seed_dir / "ace_stratified"
        )
        seed_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        seed_rows: list[pd.DataFrame] = []

        for split in SPLITS:
            source_path = (
                seed_dir
                / "chun_10ppm"
                / f"{split}_metrics.csv"
            )

            if not source_path.is_file():
                raise FileNotFoundError(
                    f"缺少ACE分层结果：{source_path}"
                )

            dataframe = pd.read_csv(source_path)

            missing = (
                REQUIRED_COLUMNS
                - set(dataframe.columns)
            )

            if missing:
                raise RuntimeError(
                    f"{source_path}缺少字段："
                    f"{sorted(missing)}"
                )

            current = dataframe.copy()
            current.insert(0, "seed", seed)
            current.insert(1, "split", split)

            seed_rows.append(current)
            all_rows.append(current)

        seed_result = pd.concat(
            seed_rows,
            ignore_index=True,
        )

        seed_csv_path = (
            seed_output_dir
            / "ace_stratified_metrics.csv"
        )
        seed_result.to_csv(
            seed_csv_path,
            index=False,
        )

        seed_json_path = (
            seed_output_dir
            / "ace_stratified_metrics.json"
        )
        seed_json_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "source": "locked CHUN evaluation",
                    "retrained": False,
                    "rows": to_json_records(
                        seed_result
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"[seed {seed}] "
            f"{seed_csv_path}"
        )

    raw = pd.concat(
        all_rows,
        ignore_index=True,
    )

    raw_path = (
        root_output_dir
        / "three_seed_raw.csv"
    )
    raw.to_csv(
        raw_path,
        index=False,
    )

    summary_rows: list[dict[str, Any]] = []

    for (
        split,
        ce_bucket,
    ), group in raw.groupby(
        ["split", "ce_bucket"],
        sort=False,
    ):
        spec_counts = sorted(
            set(
                int(value)
                for value in group[
                    "spec_count"
                ].tolist()
            )
        )

        if len(spec_counts) != 1:
            raise RuntimeError(
                f"{split}/{ce_bucket}在不同seed中"
                f"spec_count不一致：{spec_counts}"
            )

        row: dict[str, Any] = {
            "split": split,
            "ce_bucket": ce_bucket,
            "spec_count": spec_counts[0],
            "mean_ce": float(
                group["mean_ce"].mean()
            ),
            "seed_count": int(
                group["seed"].nunique()
            ),
        }

        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{metric}_mean"] = float(
                values.mean()
            )
            row[f"{metric}_std"] = float(
                values.std(ddof=1)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        root_output_dir
        / "three_seed_summary.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
    )

    summary_json_path = (
        root_output_dir
        / "three_seed_summary.json"
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "experiment_root": str(
                    experiment_root
                ),
                "seeds": list(SEEDS),
                "retrained": False,
                "summary": to_json_records(
                    summary
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        experiment_root.name.upper(),
        "ACE-STRATIFIED THREE-SEED SUMMARY",
    )
    print("=" * 88)
    print(summary.to_string(index=False))
    print()
    print("RAW:", raw_path)
    print("SUMMARY:", summary_path)


if __name__ == "__main__":
    main()
