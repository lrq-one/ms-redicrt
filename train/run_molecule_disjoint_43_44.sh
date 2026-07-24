#!/usr/bin/env bash

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"

EXPERIMENT_ROOT="$ROOT/runs/experiments/molecule_disjoint_3seeds"

SEEDS=(
    43
    44
)

run_logged() {
    LABEL="$1"
    LOG_FILE="$2"
    shift 2

    echo
    echo "================================================================================================"
    echo "$LABEL"
    echo "================================================================================================"
    echo "LOG: $LOG_FILE"
    echo

    "$@" 2>&1 | tee "$LOG_FILE"

    CODE=${PIPESTATUS[0]}

    echo
    echo "[$LABEL] code=$CODE"

    return "$CODE"
}


archive_active_outputs() {
    ARCHIVE_ROOT="$1"

    mkdir -p "$ARCHIVE_ROOT"

    for NAME in \
        v2a_gine_cutchem_only \
        v2c_ce_trajectory_ablation \
        v2e_full_063
    do
        PATH_TO_MOVE="$ROOT/runs/$NAME"

        if [ -e "$PATH_TO_MOVE" ]; then
            echo "[ARCHIVE ACTIVE] $PATH_TO_MOVE"

            mv \
                "$PATH_TO_MOVE" \
                "$ARCHIVE_ROOT/"
        fi
    done
}


save_successful_outputs() {
    SEED_DIR="$1"

    for NAME in \
        v2a_gine_cutchem_only \
        v2c_ce_trajectory_ablation \
        v2e_full_063
    do
        SOURCE="$ROOT/runs/$NAME"

        if [ ! -e "$SOURCE" ]; then
            echo "[ERROR] 缺少输出：$SOURCE"
            return 1
        fi

        mv \
            "$SOURCE" \
            "$SEED_DIR/"
    done

    return 0
}


ensure_molecule_aggregates() {
    RESULT_ROOT="$1"

    python - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path


out = Path(sys.argv[1])

base_path = (
    out
    / "final_evaluation.json"
)

if not base_path.is_file():
    raise FileNotFoundError(
        base_path
    )

result = json.loads(
    base_path.read_text(
        encoding="utf-8"
    )
)

combined_path = (
    out
    / (
        "final_evaluation_with_"
        "molecule_aggregates.json"
    )
)

if (
    "test_aggregations" in result
    and "validation_aggregations" in result
):
    combined_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "AGGREGATES_ALREADY_IN_EVALUATION"
    )
    print(
        "WROTE:",
        combined_path,
    )

    raise SystemExit(0)


validation_path = (
    out
    / "validation_molecule_aggregates.json"
)

test_path = (
    out
    / "test_molecule_aggregates.json"
)

if not validation_path.is_file():
    raise FileNotFoundError(
        validation_path
    )

if not test_path.is_file():
    raise FileNotFoundError(
        test_path
    )

result["validation_aggregations"] = (
    json.loads(
        validation_path.read_text(
            encoding="utf-8"
        )
    )
)

result["test_aggregations"] = (
    json.loads(
        test_path.read_text(
            encoding="utf-8"
        )
    )
)

result[
    "aggregation_provenance"
] = {
    "aggregation_only": True,
    "model_rerun_required": False,
    "test_used_for_selection": False,
}

combined_path.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(
    "WROTE:",
    combined_path,
)
PY

    return $?
}


write_seed_result() {
    SEED="$1"
    RESULT_PATH="$2"
    OUTPUT_PATH="$3"

    python - \
        "$SEED" \
        "$RESULT_PATH" \
        "$OUTPUT_PATH" <<'PY'
import json
import sys
from pathlib import Path


seed = int(sys.argv[1])
result_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

result = json.loads(
    result_path.read_text(
        encoding="utf-8"
    )
)

test = result["test"]
aggregates = result[
    "test_aggregations"
]
macro = aggregates[
    "molecule_macro"
]

allocator_seed = int(
    result[
        "effective_allocator_arguments"
    ][
        "seed"
    ]
)

if allocator_seed != seed:
    raise RuntimeError(
        "R184B seed不一致："
        f"{allocator_seed} != {seed}"
    )

summary = {
    "seed":
        seed,

    "micro_cosine":
        float(test["cosine"]),

    "micro_jss":
        float(test["jss"]),

    "macro_cosine":
        float(macro["cosine"]),

    "macro_jss":
        float(macro["jss"]),

    "test_spectrum_count":
        int(test["spectrum_count"]),

    "test_molecule_count":
        int(macro["molecule_count"]),

    "test_used_for_selection":
        bool(
            result["selection"][
                "test_used_for_selection"
            ]
        ),

    "allocator_seed":
        allocator_seed,
}

if summary[
    "test_used_for_selection"
]:
    raise RuntimeError(
        "检测到test参与选择"
    )

output_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print(
    "SEED RESULT:",
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
)
PY

    return $?
}


summarize_three_seeds() {
    python - "$EXPERIMENT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd


root = Path(sys.argv[1])

rows = []

for seed in (
    42,
    43,
    44,
):
    path = (
        root
        / f"seed_{seed}"
        / "seed_result.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    rows.append(
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )

raw = pd.DataFrame(rows)

raw_path = (
    root
    / "three_seed_raw_metrics.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

metrics = [
    "micro_cosine",
    "micro_jss",
    "macro_cosine",
    "macro_jss",
]

summary_rows = []

for metric in metrics:
    values = raw[metric].astype(float)

    summary_rows.append(
        {
            "metric":
                metric,

            "mean":
                float(values.mean()),

            "std_sample":
                float(
                    values.std(
                        ddof=1
                    )
                ),

            "minimum":
                float(values.min()),

            "maximum":
                float(values.max()),

            "n_seeds":
                int(len(values)),
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    root
    / "three_seed_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print()
print("=" * 100)
print("MOLECULE-DISJOINT THREE-SEED RAW RESULTS")
print("=" * 100)
print(raw.to_string(index=False))

print()
print("=" * 100)
print("MOLECULE-DISJOINT THREE-SEED SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print()
print("WROTE:", raw_path)
print("WROTE:", summary_path)
PY

    return $?
}


main() {
    cd "$ROOT" || return 1

    mkdir -p "$EXPERIMENT_ROOT"

    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT"

    for SEED in "${SEEDS[@]}"
    do
        echo
        echo
        echo "################################################################################################"
        echo "START COMPLETE PIPELINE SEED $SEED"
        echo "################################################################################################"

        SEED_DIR="$EXPERIMENT_ROOT/seed_${SEED}"

        if [ -e "$SEED_DIR" ]; then
            BACKUP="${SEED_DIR}.bak_$(date +%Y%m%d_%H%M%S)"

            echo "[BACKUP SEED DIR]"
            echo "$SEED_DIR"
            echo "→ $BACKUP"

            mv \
                "$SEED_DIR" \
                "$BACKUP"
        fi

        mkdir -p \
            "$SEED_DIR/logs"

        STALE_DIR="$EXPERIMENT_ROOT/unassigned_before_seed_${SEED}_$(date +%Y%m%d_%H%M%S)"

        archive_active_outputs \
            "$STALE_DIR"

        if [ -z "$(find "$STALE_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
            rmdir "$STALE_DIR" 2>/dev/null
        fi

        export MS2_GLOBAL_SEED="$SEED"
        export PYTHONHASHSEED="$SEED"

        printf '%s\n' \
            "MS2_GLOBAL_SEED=$MS2_GLOBAL_SEED" \
            "PYTHONHASHSEED=$PYTHONHASHSEED" \
            > "$SEED_DIR/effective_seed.env"

        run_logged \
            "SEED_${SEED}_V2A_R119" \
            "$SEED_DIR/logs/01_v2a_r119.log" \
            python -u \
            train/train.py \
            base

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED V2A/R119失败"
            return "$CODE"
        fi

        run_logged \
            "SEED_${SEED}_V2C" \
            "$SEED_DIR/logs/02_v2c.log" \
            python -u \
            train/train.py \
            control

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED V2C失败"
            return "$CODE"
        fi

        python - "$SEED" <<'PY'
import sys
from pathlib import Path

import yaml


expected = int(sys.argv[1])

paths = [
    Path(
        "runs/v2a_gine_cutchem_only/"
        "stage1_v1_40ep/config.yml"
    ),
    Path(
        "runs/v2a_gine_cutchem_only/"
        "stage2_r119_10ep/config.yml"
    ),
    Path(
        "runs/v2c_ce_trajectory_ablation/"
        "control/config.yml"
    ),
]

for path in paths:
    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    actual = int(
        config["seed"]
    )

    print(
        path,
        "seed=",
        actual,
    )

    if actual != expected:
        raise RuntimeError(
            f"{path} seed不一致："
            f"{actual} != {expected}"
        )

print(
    "BASE_R119_V2C_SEED_AUDIT_OK"
)
PY

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed配置审计失败"
            return "$CODE"
        fi

        run_logged \
            "SEED_${SEED}_REFINEMENT" \
            "$SEED_DIR/logs/03_refinement.log" \
            bash \
            train/_impl/run_refinement.sh \
            --fresh

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED refinement失败"
            return "$CODE"
        fi

        if ! grep -q \
            "R172_SEED=$SEED" \
            "runs/v2e_full_063/effective_seed.env"
        then
            echo "[STOP] R172/R184 seed审计失败"
            cat \
                "runs/v2e_full_063/effective_seed.env"
            return 1
        fi

        run_logged \
            "SEED_${SEED}_LOCKED_EVALUATION" \
            "$SEED_DIR/logs/04_locked_evaluation.log" \
            python -u \
            test/evaluate.py

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED锁定评估失败"
            return "$CODE"
        fi

        RESULT_ROOT="$ROOT/runs/v2e_full_063/final_locked_evaluation"

        ensure_molecule_aggregates \
            "$RESULT_ROOT"

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED molecule聚合失败"
            return "$CODE"
        fi

        RESULT_PATH="$RESULT_ROOT/final_evaluation_with_molecule_aggregates.json"

        if [ ! -f "$RESULT_PATH" ]; then
            echo "[STOP] 缺少最终结果：$RESULT_PATH"
            return 1
        fi

        TEMP_RESULT="$SEED_DIR/seed_result.before_move.json"

        write_seed_result \
            "$SEED" \
            "$RESULT_PATH" \
            "$TEMP_RESULT"

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED结果审计失败"
            return "$CODE"
        fi

        save_successful_outputs \
            "$SEED_DIR"

        CODE=$?

        if [ "$CODE" -ne 0 ]; then
            echo "[STOP] seed $SEED输出归档失败"
            return "$CODE"
        fi

        mv \
            "$TEMP_RESULT" \
            "$SEED_DIR/seed_result.json"

        echo
        echo "################################################################################################"
        echo "COMPLETE PIPELINE SEED $SEED FINISHED"
        echo "RESULT: $SEED_DIR/seed_result.json"
        echo "################################################################################################"
    done

    summarize_three_seeds

    CODE=$?

    if [ "$CODE" -ne 0 ]; then
        echo "[STOP] 三种子汇总失败"
        return "$CODE"
    fi

    echo
    echo "================================================================================================"
    echo "ALL THREE MOLECULE-DISJOINT RUNS FINISHED"
    echo "================================================================================================"
    echo "$EXPERIMENT_ROOT/three_seed_raw_metrics.csv"
    echo "$EXPERIMENT_ROOT/three_seed_summary.csv"
    echo "================================================================================================"

    return 0
}


main "$@"
