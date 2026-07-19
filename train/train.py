from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
_SCRIPT_DIR = _Path(__file__).resolve().parent

sys.path = [
    item
    for item in sys.path
    if _Path(item or ".").resolve() != _SCRIPT_DIR
]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from train._impl.config_builder import (
    materialize_training_config,
)


CONFIG_BUNDLE = (
    ROOT / "config/train.yml"
)

RUNTIME_CONFIG_DIR = (
    ROOT / "runs/_config"
)

BASE_TRAINING = (
    ROOT / "train/_impl/base_training.py"
)

CONTROL_FINETUNING = (
    ROOT / "train/_impl/control_finetuning.py"
)

REFINEMENT = (
    ROOT / "train/_impl/run_refinement.sh"
)

EVALUATION = (
    ROOT / "test/evaluate.py"
)


def prepare_runtime_config() -> None:
    paths = materialize_training_config(
        CONFIG_BUNDLE,
        RUNTIME_CONFIG_DIR,
    )

    required = {
        "template",
        "base_stage",
        "continuation_stage",
    }

    missing = required.difference(
        paths
    )

    if missing:
        raise RuntimeError(
            "运行时配置生成不完整："
            + ", ".join(
                sorted(missing)
            )
        )


def run_python(
    path: Path,
    *arguments: str,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(path),
            *arguments,
        ],
        cwd=ROOT,
        check=True,
    )


def run_shell(
    path: Path,
    *arguments: str,
) -> None:
    subprocess.run(
        [
            "bash",
            str(path),
            *arguments,
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MS2 spectrum prediction "
            "mainline pipeline"
        )
    )

    parser.add_argument(
        "stage",
        choices=[
            "base",
            "control",
            "refinement",
            "evaluation",
            "all",
        ],
    )

    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    prepare_runtime_config()

    if args.stage in {
        "base",
        "all",
    }:
        run_python(
            BASE_TRAINING
        )

    if args.stage in {
        "control",
        "all",
    }:
        run_python(
            CONTROL_FINETUNING
        )

    if args.stage in {
        "refinement",
        "all",
    }:
        run_shell(
            REFINEMENT,
            *args.extra,
        )

    if args.stage in {
        "evaluation",
        "all",
    }:
        run_python(
            EVALUATION
        )


if __name__ == "__main__":
    main()
