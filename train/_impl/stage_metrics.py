from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def global_cosine(path: str | Path) -> float:
    frame = pd.read_csv(path)

    if "ce_bucket" in frame.columns:
        global_rows = frame[
            frame["ce_bucket"].astype(str) == "global"
        ]

        if len(global_rows):
            return float(global_rows.iloc[0]["cos"])

    if "spec_count" in frame.columns:
        denominator = float(frame["spec_count"].sum())

        if denominator <= 0:
            raise RuntimeError(f"spec_count总和无效：{path}")

        return float(
            (
                frame["cos"]
                * frame["spec_count"]
            ).sum()
            / denominator
        )

    if "global_cos" in frame.columns:
        return float(frame.iloc[0]["global_cos"])

    if "cos" in frame.columns and len(frame) == 1:
        return float(frame.iloc[0]["cos"])

    raise RuntimeError(
        f"无法从{path}提取global cosine；"
        f"columns={list(frame.columns)}"
    )


def select_stage(args: argparse.Namespace) -> None:
    before = global_cosine(args.before)
    best = global_cosine(args.best)

    use_child = best > before + float(args.min_delta)
    selected = args.child if use_child else args.parent

    result = {
        "before_cosine": before,
        "best_cosine": best,
        "delta": best - before,
        "min_delta": float(args.min_delta),
        "accepted": use_child,
        "selected_checkpoint": selected,
    }

    Path(args.decision).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(selected)


def best_alpha(path: str | Path) -> float:
    frame = pd.read_csv(path)

    if "val_cos" not in frame.columns:
        raise RuntimeError(
            f"alpha表缺少val_cos：{path}"
        )

    row = (
        frame.sort_values(
            "val_cos",
            ascending=False,
        )
        .iloc[0]
    )

    return float(row["alpha"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    metric_parser = sub.add_parser("metric")
    metric_parser.add_argument("path")

    alpha_parser = sub.add_parser("alpha")
    alpha_parser.add_argument("path")

    select_parser = sub.add_parser("select")
    select_parser.add_argument("--before", required=True)
    select_parser.add_argument("--best", required=True)
    select_parser.add_argument("--parent", required=True)
    select_parser.add_argument("--child", required=True)
    select_parser.add_argument("--decision", required=True)
    select_parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    if args.command == "metric":
        print(f"{global_cosine(args.path):.12f}")
    elif args.command == "alpha":
        print(f"{best_alpha(args.path):.12g}")
    elif args.command == "select":
        select_stage(args)


if __name__ == "__main__":
    main()
