from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path.cwd()

RANDOM_ROOT = (
    ROOT
    / "runs"
    / "experiments"
    / "molecule_disjoint_3seeds"
)

SCAFFOLD_ROOT = (
    ROOT
    / "runs"
    / "experiments"
    / "scaffold_disjoint_3seeds"
)

REPORT_PATHS = {
    "random_retrieval": (
        RANDOM_ROOT
        / "candidate_retrieval"
        / "preparation"
        / "input_inventory.json"
    ),
    "scaffold_retrieval": (
        SCAFFOLD_ROOT
        / "candidate_retrieval"
        / "preparation"
        / "input_inventory.json"
    ),
    "random_formula": (
        RANDOM_ROOT
        / "formula_annotations"
        / "preparation"
        / "input_inventory.json"
    ),
}

for path in REPORT_PATHS.values():
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

for experiment_root in (
    RANDOM_ROOT,
    SCAFFOLD_ROOT,
):
    for seed in (42, 43, 44):
        (
            experiment_root
            / f"seed_{seed}"
            / "candidate_retrieval"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

for seed in (42, 43, 44):
    (
        RANDOM_ROOT
        / f"seed_{seed}"
        / "formula_annotations"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def unique_paths(
    paths: list[Path],
) -> list[Path]:
    result = []
    seen = set()

    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        key = str(resolved)

        if key in seen:
            continue

        if not resolved.exists():
            continue

        seen.add(key)
        result.append(resolved)

    return sorted(
        result,
        key=lambda item: str(item),
    )


def path_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return -1


def human_size(size: int) -> str:
    if size < 0:
        return "unknown"

    value = float(size)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


def serialise_value(
    value: Any,
    limit: int = 300,
) -> str:
    text = repr(value)

    if len(text) > limit:
        text = text[:limit] + "..."

    return text


def dataframe_info(
    path: Path,
    load_limit_bytes: int = 800_000_000,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path_size(path),
        "size_human": human_size(
            path_size(path)
        ),
        "loaded": False,
    }

    size = path_size(path)

    if (
        size < 0
        or size > load_limit_bytes
    ):
        result["load_skipped_reason"] = (
            "file larger than audit limit"
        )
        return result

    name = path.name.lower()

    try:
        if (
            name.endswith(".pkl")
            or name.endswith(".pkl.gz")
            or name.endswith(".pickle")
            or name.endswith(".pickle.gz")
        ):
            dataframe = pd.read_pickle(path)

        elif (
            name.endswith(".csv")
            or name.endswith(".csv.gz")
        ):
            dataframe = pd.read_csv(path)

        elif name.endswith(".parquet"):
            dataframe = pd.read_parquet(path)

        else:
            result["load_skipped_reason"] = (
                "unsupported extension"
            )
            return result

    except Exception as error:
        result["load_error"] = (
            f"{type(error).__name__}: {error}"
        )
        return result

    result["loaded"] = True
    result["python_type"] = str(
        type(dataframe)
    )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        result["preview"] = serialise_value(
            dataframe
        )
        return result

    result["shape"] = [
        int(dataframe.shape[0]),
        int(dataframe.shape[1]),
    ]

    result["index_name"] = (
        None
        if dataframe.index.name is None
        else str(dataframe.index.name)
    )

    result["columns"] = [
        str(column)
        for column in dataframe.columns
    ]

    result["dtypes"] = {
        str(column): str(dtype)
        for column, dtype
        in dataframe.dtypes.items()
    }

    samples = {}

    for column in dataframe.columns:
        series = dataframe[column].dropna()

        if series.empty:
            continue

        samples[str(column)] = {
            "sample_type": str(
                type(series.iloc[0])
            ),
            "sample_value": serialise_value(
                series.iloc[0]
            ),
        }

    result["column_samples"] = samples

    return result


def find_dirs_with_files(
    roots: list[Path],
    required_names: tuple[str, ...],
) -> list[Path]:
    candidates = []

    for root in roots:
        if not root.is_dir():
            continue

        for first_name in required_names[:1]:
            for path in root.rglob(first_name):
                parent = path.parent

                if all(
                    (parent / filename).is_file()
                    for filename in required_names
                ):
                    candidates.append(parent)

    return unique_paths(candidates)


search_roots = unique_paths(
    [
        ROOT / "data",
        ROOT.parent / "data",
        ROOT / "baseline",
        ROOT.parent / "baseline",
    ]
)

proc_dirs = find_dirs_with_files(
    search_roots,
    (
        "spec_df.pkl",
        "mol_df.pkl",
    ),
)

split_dirs = find_dirs_with_files(
    search_roots,
    (
        "train_ids.csv",
        "val_ids.csv",
        "test_ids.csv",
    ),
)

all_candidate_files: list[Path] = []
all_annotation_files: list[Path] = []

candidate_patterns = (
    "*candidate*.pkl",
    "*candidate*.pkl.gz",
    "*candidate*.csv",
    "*candidate*.csv.gz",
    "*candidate*.parquet",
    "*pubchem*.pkl",
    "*pubchem*.pkl.gz",
    "*pubchem*.csv",
    "*pubchem*.csv.gz",
    "*pubchem*.parquet",
)

annotation_patterns = (
    "ann_df*.pkl",
    "ann_df*.pkl.gz",
    "*annotation*.pkl",
    "*annotation*.pkl.gz",
    "*annotations*.pkl",
    "*annotations*.pkl.gz",
    "*magma*ann*.pkl",
    "*magma*ann*.pkl.gz",
)

for root in search_roots:
    if not root.is_dir():
        continue

    for pattern in candidate_patterns:
        all_candidate_files.extend(
            root.rglob(pattern)
        )

    for pattern in annotation_patterns:
        all_annotation_files.extend(
            root.rglob(pattern)
        )

all_candidate_files = unique_paths(
    all_candidate_files
)

all_annotation_files = unique_paths(
    all_annotation_files
)

proc_reports = []

for directory in proc_dirs:
    proc_report = {
        "directory": str(directory),
        "spec_df": dataframe_info(
            directory / "spec_df.pkl"
        ),
        "mol_df": dataframe_info(
            directory / "mol_df.pkl"
        ),
    }

    for annotation_name in (
        "ann_df.pkl",
        "ann_df.pkl.gz",
    ):
        annotation_path = (
            directory / annotation_name
        )

        if annotation_path.is_file():
            proc_report[
                annotation_name
            ] = dataframe_info(
                annotation_path
            )

    proc_reports.append(proc_report)

split_reports = []

for directory in split_dirs:
    current = {
        "directory": str(directory),
    }

    for split in (
        "train",
        "val",
        "test",
    ):
        current[split] = dataframe_info(
            directory
            / f"{split}_ids.csv"
        )

    split_reports.append(current)

candidate_reports = [
    dataframe_info(path)
    for path in all_candidate_files
]

annotation_reports = []

for path in all_annotation_files:
    lower_name = path.name.lower()

    if (
        "magma" in lower_name
        or "graff" in lower_name
    ):
        ground_truth_status = (
            "model-derived annotation; "
            "do not use as expert formula ground truth"
        )
    else:
        ground_truth_status = (
            "candidate annotation source; "
            "must verify provenance and columns"
        )

    report = dataframe_info(path)
    report[
        "ground_truth_status"
    ] = ground_truth_status
    annotation_reports.append(report)

common_report = {
    "project_root": str(ROOT),
    "search_roots": [
        str(path)
        for path in search_roots
    ],
    "proc_datasets": proc_reports,
    "split_datasets": split_reports,
    "candidate_or_pubchem_files": (
        candidate_reports
    ),
    "annotation_files": (
        annotation_reports
    ),
    "protocol": {
        "retrieval_candidate_count": 50,
        "true_candidate_count": 1,
        "hard_negative_count": 49,
        "mass_tolerance_ppm": 10.0,
        "fingerprint": "Morgan radius 2",
        "ranking": (
            "descending Tanimoto similarity"
        ),
        "candidate_pool_scope": (
            "one frozen pool per target molecule; "
            "shared across all ACE spectra, seeds, "
            "and models within the same split"
        ),
        "retrieval_splits": [
            "molecule_disjoint",
            "scaffold_disjoint",
        ],
        "formula_annotation_split": (
            "molecule_disjoint only"
        ),
        "formula_ground_truth": (
            "NIST expert formula annotations only"
        ),
        "forbidden_formula_ground_truth": (
            "MAGMa or model-generated annotations"
        ),
    },
}

for name, report_path in REPORT_PATHS.items():
    report = dict(common_report)
    report["report_type"] = name

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
print("=" * 88)
print("DOWNSTREAM INPUT AUDIT")
print("=" * 88)

print("\nPROC DATASETS:")
for item in proc_reports:
    print(" ", item["directory"])

print("\nSPLIT DATASETS:")
for item in split_reports:
    print(" ", item["directory"])

print("\nPUBCHEM / CANDIDATE FILES:")
if candidate_reports:
    for item in candidate_reports:
        print(
            " ",
            item["path"],
            item["size_human"],
            "loaded=",
            item["loaded"],
        )
else:
    print("  NONE FOUND")

print("\nANNOTATION FILES:")
if annotation_reports:
    for item in annotation_reports:
        print(
            " ",
            item["path"],
            item["size_human"],
        )
        print(
            "    ",
            item["ground_truth_status"],
        )
else:
    print("  NONE FOUND")

print("\nREPORTS:")
for report_path in REPORT_PATHS.values():
    print(" ", report_path)
