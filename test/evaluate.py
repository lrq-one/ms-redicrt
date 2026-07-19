from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


# Only suppress the known sklearn/LightGBM feature-name warning.
# Other warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=(
        r"X does not have valid feature names, "
        r"but LGBMRegressor was fitted with feature names"
    ),
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


ROOT = Path(__file__).resolve().parents[1]

RUN_ROOT = ROOT / "runs/v2e_full_063"

OUTPUT_DIR = (
    RUN_ROOT
    / "final_locked_evaluation"
)

DIAGNOSTICS = (
    ROOT
    / "train/_impl/refinement_steps"
)

TEMPLATE_PATH = (
    ROOT / "runs/_config/template.yml"
)

CONFIG_PATH = (
    ROOT
    / "runs/v2c_ce_trajectory_ablation/"
    "control/config.yml"
)

BACKBONE_PATH = (
    RUN_ROOT
    / "08_R160/r160_best_state.pt"
)

RERANKER_PATH = (
    RUN_ROOT
    / "09_R172D/r170_regressor.pkl"
)

ALLOCATOR_PATH = (
    RUN_ROOT
    / "11_R184B/r184_allocator_best.pt"
)

RERANKER_SCRIPT = (
    DIAGNOSTICS
    / "candidate_reranker.py"
)

ALLOCATOR_SCRIPT = (
    DIAGNOSTICS
    / "spectrum_allocator.py"
)

VALIDATION_REFERENCE = 0.6359706549675825
PARITY_TOLERANCE = 1.0e-4
SEED = 3407


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def load_module(
    path: Path,
    name: str,
) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
            type(None),
        ),
    ):
        return value

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
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)


def global_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    rows = frame[
        frame["ce_bucket"].astype(str)
        == "global"
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one global row."
        )

    row = rows.iloc[0]

    return {
        "cosine": float(row["cos"]),
        "jss": float(row["jss"]),
        "spectrum_count": int(
            row["spec_count"]
        ),
        "mean_ce": float(row["mean_ce"]),
    }


required_paths = [
    TEMPLATE_PATH,
    CONFIG_PATH,
    BACKBONE_PATH,
    RERANKER_PATH,
    ALLOCATOR_PATH,
    RERANKER_SCRIPT,
    ALLOCATOR_SCRIPT,
]

for required_path in required_paths:
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Missing required file: "
            f"{required_path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

seed_everything(SEED)

candidate_reranker = load_module(
    RERANKER_SCRIPT,
    "locked_candidate_reranker",
)

spectrum_allocator = load_module(
    ALLOCATOR_SCRIPT,
    "locked_spectrum_allocator",
)


# -------------------------------------------------------------------------
# Eliminate the sklearn warning correctly:
# when the LightGBM model stores feature names, supply a DataFrame with the
# same names. The warning filter above is only a fallback.
# -------------------------------------------------------------------------

def predict_with_feature_names(
    regressor: Any,
    features: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    score_clip: float,
) -> torch.Tensor:
    array = (
        features
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    feature_names = getattr(
        regressor,
        "feature_names_in_",
        None,
    )

    if feature_names is None:
        feature_names = getattr(
            regressor,
            "feature_name_",
            None,
        )

    predict_input: Any = array

    if feature_names is not None:
        names = [
            str(name)
            for name in list(feature_names)
        ]

        if len(names) == int(array.shape[1]):
            predict_input = pd.DataFrame(
                array,
                columns=names,
            )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"X does not have valid feature names, "
                r"but LGBMRegressor was fitted "
                r"with feature names"
            ),
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )

        score = (
            regressor
            .predict(predict_input)
            .astype(np.float32)
        )

    score = np.clip(
        score,
        -float(score_clip),
        float(score_clip),
    )

    return torch.from_numpy(
        score
    ).to(
        device=device,
        dtype=dtype,
    )


spectrum_allocator.lgbm_predict = (
    predict_with_feature_names
)


from ms2spectra.workflow import (
    load_config,
    init_dataset,
    init_dataloader,
)

from ms2spectra.training import FragGNNPL


with RERANKER_PATH.open(
    "rb"
) as handle:
    reranker_package = pickle.load(
        handle
    )

regressor = reranker_package["model"]

allocator_package = torch.load(
    ALLOCATOR_PATH,
    map_location="cpu",
    weights_only=False,
)

saved_arguments = dict(
    allocator_package["args"]
)

allocator_arguments = SimpleNamespace(
    **saved_arguments
)

extra_schema = allocator_package.get(
    "extra_schema",
    reranker_package.get(
        "extra_schema",
        [],
    ),
)

saved_validation_cosine = float(
    allocator_package.get(
        "best_val_cos",
        float("nan"),
    )
)

config = load_config(
    TEMPLATE_PATH,
    CONFIG_PATH,
)

config = candidate_reranker.force_r160_arch(
    config
)

validation_dataset, test_dataset = (
    init_dataset(
        config,
        splits=(
            "val",
            "test",
        ),
    )
)

validation_loader = init_dataloader(
    validation_dataset,
    config,
)

test_loader = init_dataloader(
    test_dataset,
    config,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

backbone = FragGNNPL(
    **config
)

backbone_state = (
    candidate_reranker
    .load_state_dict_any(
        BACKBONE_PATH
    )
)

missing, unexpected = (
    backbone.load_state_dict(
        backbone_state,
        strict=False,
    )
)

print(
    "Backbone missing keys:",
    len(missing),
)

print(
    "Backbone unexpected keys:",
    len(unexpected),
)

backbone = backbone.to(device)
backbone.eval()

for parameter in backbone.parameters():
    parameter.requires_grad_(False)

allocator = (
    spectrum_allocator
    .ResidualAllocator(
        input_dim=int(
            allocator_package["input_dim"]
        ),
        hidden=int(
            saved_arguments["hidden"]
        ),
        layers=int(
            saved_arguments["layers"]
        ),
        dropout=float(
            saved_arguments["dropout"]
        ),
        score_clip=float(
            saved_arguments["score_clip"]
        ),
    )
    .to(device)
)

allocator.load_state_dict(
    allocator_package["model"]
)

allocator.eval()

print()
print("=" * 100)
print("LOCKED EFFECTIVE EVALUATION SETTINGS")
print("=" * 100)

for key in (
    "alpha",
    "lgbm_score_clip",
    "residual_scale",
    "temperature",
    "eval_bin_res",
    "target_bin_res",
    "local_bin_res",
    "max_extra_dims",
):
    print(
        f"{key}:",
        saved_arguments.get(key),
    )

print(
    "input_dim:",
    allocator_package["input_dim"],
)

print(
    "extra_schema:",
    extra_schema,
)

print("=" * 100)


# -------------------------------------------------------------------------
# Validation parity check
# -------------------------------------------------------------------------

seed_everything(SEED)

validation_table, validation_detail = (
    spectrum_allocator.eval_split(
        base=backbone,
        allocator=allocator,
        regressor=regressor,
        extra_schema=extra_schema,
        dl=validation_loader,
        device=device,
        r170=candidate_reranker,
        args=allocator_arguments,
        split="val",
    )
)

validation_table.to_csv(
    OUTPUT_DIR
    / "validation_metrics.csv",
    index=False,
)

validation_detail.to_csv(
    OUTPUT_DIR
    / "validation_per_spectrum.csv",
    index=False,
)

validation_metrics = global_metrics(
    validation_table
)

validation_difference = (
    validation_metrics["cosine"]
    - VALIDATION_REFERENCE
)

package_difference = (
    validation_metrics["cosine"]
    - saved_validation_cosine
)

validation_parity = (
    abs(validation_difference)
    <= PARITY_TOLERANCE
)

print()
print("=" * 100)
print("LOCKED VALIDATION RESULT")
print("=" * 100)
print(validation_table.to_string(index=False))
print()
print(
    "Reference validation:",
    VALIDATION_REFERENCE,
)
print(
    "Saved package validation:",
    saved_validation_cosine,
)
print(
    "Recomputed validation:",
    validation_metrics["cosine"],
)
print(
    "Difference vs reference:",
    validation_difference,
)
print(
    "Parity tolerance:",
    PARITY_TOLERANCE,
)
print(
    "Validation parity:",
    validation_parity,
)
print("=" * 100)

if not validation_parity:
    raise RuntimeError(
        "Validation parity failed. "
        "Test evaluation was not started."
    )


# -------------------------------------------------------------------------
# Exact test evaluation using the same evaluator and saved model arguments
# -------------------------------------------------------------------------

seed_everything(SEED)

test_table, test_detail = (
    spectrum_allocator.eval_split(
        base=backbone,
        allocator=allocator,
        regressor=regressor,
        extra_schema=extra_schema,
        dl=test_loader,
        device=device,
        r170=candidate_reranker,
        args=allocator_arguments,
        split="test",
    )
)

test_table.to_csv(
    OUTPUT_DIR
    / "test_metrics.csv",
    index=False,
)

test_detail.to_csv(
    OUTPUT_DIR
    / "test_per_spectrum.csv",
    index=False,
)

test_metrics = global_metrics(
    test_table
)

result = {
    "experiment":
        "locked_final_model_evaluation",

    "status":
        (
            "accepted"
            if (
                validation_parity
                and test_metrics["cosine"]
                >= 0.65
            )
            else "below_target"
        ),

    "selection": {
        "selected_on":
            "validation",

        "test_used_for_selection":
            False,

        "final_model":
            "strong_spectrum_allocator",

        "validation_reference":
            VALIDATION_REFERENCE,

        "saved_package_validation":
            saved_validation_cosine,

        "recomputed_validation":
            validation_metrics["cosine"],

        "validation_parity_tolerance":
            PARITY_TOLERANCE,

        "validation_parity":
            validation_parity,
    },

    "validation":
        validation_metrics,

    "test":
        test_metrics,

    "effective_allocator_arguments":
        json_safe(saved_arguments),

    "extra_schema":
        json_safe(extra_schema),

    "artifact_sha256": {
        "refined_backbone":
            sha256(BACKBONE_PATH),

        "candidate_reranker":
            sha256(RERANKER_PATH),

        "spectrum_allocator":
            sha256(ALLOCATOR_PATH),

        "model_config":
            sha256(CONFIG_PATH),

        "candidate_reranker_script":
            sha256(RERANKER_SCRIPT),

        "spectrum_allocator_script":
            sha256(ALLOCATOR_SCRIPT),
    },
}

result_path = (
    OUTPUT_DIR
    / "final_evaluation.json"
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

arguments_path = (
    OUTPUT_DIR
    / "effective_allocator_arguments.json"
)

arguments_path.write_text(
    json.dumps(
        json_safe(saved_arguments),
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 100)
print("FINAL EXACT TEST RESULT")
print("=" * 100)
print(test_table.to_string(index=False))
print()
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
)
print("=" * 100)
print(
    "WROTE:",
    result_path,
)
