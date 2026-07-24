#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$ROOT/runs/v2e_full_063"
DIAG="$ROOT/train/_impl/refinement_steps"
TEMPLATE="$ROOT/runs/_config/template.yml"

BASE_CONFIG="$ROOT/runs/v2c_ce_trajectory_ablation/control/config.yml"

BASE_CHECKPOINT="$ROOT/runs/v2c_ce_trajectory_ablation/control/model_best.ckpt"

V2C_REFERENCE="0.5970845222"
V2E_REFERENCE="0.6055593451708329"

FRESH=0

if [ "${1:-}" = "--fresh" ]; then
    FRESH=1
fi

cd "$ROOT" || {
    echo "无法进入：$ROOT"
    exit 1
}

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ "$FRESH" -eq 1 ] && [ -d "$OUT" ]; then
    BACKUP="${OUT}.bak_$(date +%Y%m%d_%H%M%S)"
    mv "$OUT" "$BACKUP"
    echo "旧输出已备份：$BACKUP"
fi

mkdir -p \
    "$OUT/logs" \
    "$OUT/00_preflight" \
    "$OUT/01_R146" \
    "$OUT/02_R147" \
    "$OUT/03_R149B" \
    "$OUT/04_R150A" \
    "$OUT/05_R150B" \
    "$OUT/06_R153" \
    "$OUT/07_R154" \
    "$OUT/08_R160" \
    "$OUT/09_R172D" \
    "$OUT/11_R184B" \

if [ ! -f "$TEMPLATE" ]; then
    echo "缺少模板：$TEMPLATE"
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo "缺少V2C配置：$BASE_CONFIG"
    exit 1
fi

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "缺少V2C checkpoint：$BASE_CHECKPOINT"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "没有检测到nvidia-smi，停止。"
    exit 1
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA不可用")

print("CUDA:", torch.cuda.get_device_name(0))

try:
    import lightgbm
    print("LightGBM:", lightgbm.__version__)
except Exception as exc:
    raise SystemExit(
        f"LightGBM不可用：{exc!r}"
    )
PY

PREFLIGHT_CODE=$?

if [ "$PREFLIGHT_CODE" -ne 0 ]; then
    echo "环境检查失败。"
    exit 1
fi

python - "$ROOT/config/train.yml" \
    > "$OUT/mainline_hparams.env" \
    2> "$OUT/mainline_hparams.log" <<'PY_HPARAMS'
from pathlib import Path
import shlex
import sys
import yaml

config = yaml.safe_load(
    Path(sys.argv[1]).read_text(
        encoding="utf-8"
    )
)

params = config.get("postprocessing_env")

if not isinstance(params, dict):
    raise RuntimeError(
        "缺少postprocessing_env配置"
    )

for key, value in params.items():
    print(
        f"{key}={shlex.quote(str(value))}"
    )
PY_HPARAMS

HPARAM_CODE=$?

if [ "$HPARAM_CODE" -ne 0 ]; then
    echo "主线超参数读取失败。"
    exit 1
fi

# shellcheck disable=SC1090
source "$OUT/mainline_hparams.env"

run_stage() {
    LABEL="$1"
    shift

    LOG_FILE="$OUT/logs/${LABEL}.log"
    COMMAND_FILE="$OUT/logs/${LABEL}.command.txt"

    printf '%q ' "$@" \
        > "$COMMAND_FILE"

    printf '\n' \
        >> "$COMMAND_FILE"

    echo
    echo "================================================================================================"
    echo "$LABEL"
    echo "================================================================================================"
    cat "$COMMAND_FILE"
    echo

    "$@" 2>&1 \
        | tee "$LOG_FILE"

    CODE=${PIPESTATUS[0]}

    echo
    echo "$LABEL exit code: $CODE"

    return "$CODE"
}

metric() {
    python "$ROOT/train/_impl/stage_metrics.py" \
        metric \
        "$1"
}

echo
echo "================================================================================================"
echo "V2E-FULL-063 PIPELINE"
echo "================================================================================================"
echo "V2C reference : $V2C_REFERENCE"
echo "V2E reference : $V2E_REFERENCE"
echo "test used      : False"
echo "output         : $OUT"
echo "================================================================================================"

python - <<'PY' \
    2>&1 \
    | tee "$OUT/00_preflight/preflight.log"
from pathlib import Path

import torch

from ms2spectra.workflow import (
    load_config,
    init_dataset,
    init_dataloader,
)
from ms2spectra.training import FragGNNPL

import importlib.util


root = Path(
    "/home/lwh/projects/lrq2/"
    "fragnnet-main/ms2spectra_v1_r119"
)

script = (
    root
    / "train/_impl/refinement_steps/"
    "formula_composition.py"
)

spec = importlib.util.spec_from_file_location(
    "r146_preflight",
    str(script),
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Args:
    hidden = 128
    dropout = 0.05
    delta_scale = 0.05
    formula_comp_feat_size = 18
    bin_res = 0.01
    max_bins = 0
    ce_binned_aux_weight = 0.0015
    low_w = 0.30
    mid_w = 1.50
    high_w = 2.00
    r117_weight = 0.0
    r117_false_weight = 0.25


config = load_config(
    root / "runs/_config/template.yml",
    root
    / "runs/v2c_ce_trajectory_ablation/"
    "control/config.yml",
)

config = module.override_cfg(
    config,
    Args(),
)

train_dataset = init_dataset(
    config,
    splits=("train",),
)[0]

loader = init_dataloader(
    train_dataset,
    config,
)

model = FragGNNPL(
    **config
)

checkpoint = torch.load(
    root
    / "runs/v2c_ce_trajectory_ablation/"
    "control/model_best.ckpt",
    map_location="cpu",
    weights_only=False,
)

state_dict = (
    checkpoint["state_dict"]
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    )
    else checkpoint
)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False,
)

print("missing keys:", len(missing))
for key in missing[:40]:
    print("  missing:", key)

print("unexpected keys:", len(unexpected))
for key in unexpected[:40]:
    print("  unexpected:", key)

allowed_missing_prefixes = (
    "model.formula_comp_residual_head",
)

bad_missing = [
    key
    for key in missing
    if not key.startswith(
        allowed_missing_prefixes
    )
]

if bad_missing:
    raise RuntimeError(
        "V2C→R146存在非预期missing keys："
        + repr(
            bad_missing[:40]
        )
    )

if unexpected:
    raise RuntimeError(
        "V2C→R146存在unexpected keys："
        + repr(
            unexpected[:40]
        )
    )

batch = next(
    iter(loader)
)

device = torch.device("cuda")
model = model.to(device)
batch = module.move_to_device(
    batch,
    device,
)

model.eval()

with torch.no_grad():
    result = model._common_step(
        batch,
        split="train",
        log=False,
    )

if not torch.isfinite(
    result["mean_loss"]
):
    raise RuntimeError(
        "preflight mean_loss非有限值"
    )

print(
    "preflight mean_loss:",
    float(
        result["mean_loss"]
        .detach()
        .cpu()
    ),
)

print("V2C_TO_R146_PREFLIGHT_PASSED")
PY

PREFLIGHT_RUN=${PIPESTATUS[0]}

if [ "$PREFLIGHT_RUN" -ne 0 ]; then
    echo "V2C→R146兼容性检查失败。"
    exit 1
fi


# =============================================================================
# R146
# =============================================================================

R146_CKPT="$OUT/01_R146/r146_best_state.pt"

if [ ! -f "$R146_CKPT" ]; then
    run_stage \
        "01_R146" \
        python -u \
        "$DIAG/formula_composition.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$BASE_CHECKPOINT" \
        --out_dir "$OUT/01_R146" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-5 \
        --weight_decay 1e-5 \
        --hidden 128 \
        --dropout 0.05 \
        --delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --bin_res 0.01 \
        --max_bins 0 \
        --ce_binned_aux_weight 0.0015 \
        --r117_weight 0.0 \
        --r117_false_weight 0.25 \
        --low_w 0.30 \
        --mid_w 1.50 \
        --high_w 2.00

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R146 checkpoint已存在。"
fi


# =============================================================================
# R147
# =============================================================================

R147_CKPT="$OUT/02_R147/r147_best_state.pt"

if [ ! -f "$R147_CKPT" ]; then
    run_stage \
        "02_R147" \
        python -u \
        "$DIAG/collision_energy_response.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R146_CKPT" \
        --out_dir "$OUT/02_R147" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-5 \
        --weight_decay 1e-5 \
        --k3b_hidden 128 \
        --k3b_dropout 0.05 \
        --k3b_delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --ce_hidden 128 \
        --ce_dropout 0.05 \
        --ce_delta_scale 0.025 \
        --ce_use_formula_comp \
        --ce_use_depth \
        --ce_use_h \
        --bin_res 0.01 \
        --max_bins 0 \
        --ce_binned_aux_weight 0.0015 \
        --r117_weight 0.0 \
        --r117_false_weight 0.20 \
        --low_w 0.25 \
        --mid_w 1.75 \
        --high_w 2.25

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R147 checkpoint已存在。"
fi


FORMULA_COMMON=(
    --k3b_hidden 128
    --k3b_dropout 0.05
    --k3b_delta_scale 0.05
    --formula_comp_feat_size 18
    --ce_hidden 128
    --ce_dropout 0.05
    --ce_delta_scale 0.02
    --ce_use_formula_comp
    --ce_use_depth
    --ce_use_h
    --bin_res 0.01
    --max_bins 0
    --ce_binned_aux_weight 0.0015
    --r117_weight 0.0
    --r117_false_weight 0.20
    --low_w 0.25
    --mid_w 1.75
    --high_w 2.25
    --formula_aux_weight 0.0005
    --formula_tol 0.01
    --formula_mz_sigma 0.003
    --hard_formula_topk 3
    --formula_score_mode max
    --prob_alpha 0.0
    --formula_kl_weight 1.0
    --formula_rank_weight 0.2
    --formula_false_weight 0.01
    --formula_target_topk 5
    --formula_neg_topk 20
    --formula_margin 0.5
    --formula_neg_target_max 0.002
    --formula_low_w 0.05
    --formula_mid_w 1.5
    --formula_high_w 2.0
)


# =============================================================================
# R149B
# =============================================================================

R149_CKPT="$OUT/03_R149B/r148_best_state.pt"

if [ ! -f "$R149_CKPT" ]; then
    run_stage \
        "03_R149B" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R147_CKPT" \
        --out_dir "$OUT/03_R149B" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_k3b \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R149B checkpoint已存在。"
fi


# =============================================================================
# R150A
# =============================================================================

R150A_CKPT="$OUT/04_R150A/r148_best_state.pt"

if [ ! -f "$R150A_CKPT" ]; then
    run_stage \
        "04_R150A" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R149_CKPT" \
        --out_dir "$OUT/04_R150A" \
        --epochs 8 \
        --max_train_batches -1 \
        --lr 3e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_k3b \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R150A checkpoint已存在。"
fi


# =============================================================================
# R150B
# =============================================================================

R150B_CKPT="$OUT/05_R150B/r148_best_state.pt"

if [ ! -f "$R150B_CKPT" ]; then
    run_stage \
        "05_R150B" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R150A_CKPT" \
        --out_dir "$OUT/05_R150B" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 1e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_k3b \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R150B checkpoint已存在。"
fi


# =============================================================================
# R153 reconstructed from surviving train_frag_rep implementation
# =============================================================================

R153_CKPT="$OUT/06_R153/r148_best_state.pt"

if [ ! -f "$R153_CKPT" ]; then
    run_stage \
        "06_R153" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R150B_CKPT" \
        --out_dir "$OUT/06_R153" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 8e-7 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_k3b \
        --train_formula_module \
        --train_frag_rep

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R153 checkpoint已存在。"
fi


# =============================================================================
# R154 bounded residual flow, validation gated
# =============================================================================

R154_CKPT="$OUT/07_R154/r148_best_state.pt"

if [ ! -f "$R154_CKPT" ]; then
    run_stage \
        "07_R154" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R153_CKPT" \
        --out_dir "$OUT/07_R154" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 5e-7 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --use_ce_flowfrag \
        --ce_flowfrag_lambda_max 0.15 \
        --ce_flowfrag_hidden 128 \
        --ce_flowfrag_dropout 0.05 \
        --ce_flowfrag_max_depth 4 \
        --ce_flowfrag_mixture_hidden 128 \
        --ce_flowfrag_mixture_dropout 0.05 \
        --ce_flowfrag_mixture_init_bias -3.0 \
        --ce_flowfrag_delta_clip 3.0 \
        --ce_flowfrag_use_direct_node \
        --ce_flowfrag_direct_mix 0.35 \
        --train_ce_flowfrag

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R154 checkpoint已存在。"
fi

R154_SELECTED="$(
    python "$ROOT/train/_impl/stage_metrics.py" \
        select \
        --before "$OUT/07_R154/r148_val_epoch0_before.csv" \
        --best "$OUT/07_R154/r148_best_val.csv" \
        --parent "$R153_CKPT" \
        --child "$R154_CKPT" \
        --decision "$OUT/07_R154/validation_gate.json" \
        --min-delta 0.0
)"

echo "R154 selected checkpoint: $R154_SELECTED"


# =============================================================================
# R160
# =============================================================================

R160_CKPT="$OUT/08_R160/r160_best_state.pt"

if [ ! -f "$R160_CKPT" ]; then
    run_stage \
        "08_R160" \
        python -u \
        "$DIAG/peak_distillation.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R154_SELECTED" \
        --out_dir "$OUT/08_R160" \
        --epochs 8 \
        --max_train_batches -1 \
        --lr 2e-7 \
        --weight_decay 1e-5 \
        --k3b_hidden 128 \
        --k3b_dropout 0.05 \
        --k3b_delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --ce_hidden 128 \
        --ce_dropout 0.05 \
        --ce_delta_scale 0.020 \
        --ce_use_formula_comp \
        --ce_use_depth \
        --ce_use_h \
        --use_ce_flowfrag \
        --ce_flowfrag_lambda_max 0.15 \
        --ce_flowfrag_hidden 128 \
        --ce_flowfrag_dropout 0.05 \
        --ce_flowfrag_max_depth 4 \
        --ce_flowfrag_mixture_hidden 128 \
        --ce_flowfrag_mixture_dropout 0.05 \
        --ce_flowfrag_mixture_init_bias -3.0 \
        --ce_flowfrag_delta_clip 3.0 \
        --ce_flowfrag_use_direct_node \
        --ce_flowfrag_direct_mix 0.35 \
        --bin_res 0.01 \
        --max_bins 0 \
        --eval_bin_res 0.01 \
        --ce_binned_aux_weight 0.0015 \
        --low_w 0.25 \
        --mid_w 1.75 \
        --high_w 2.25 \
        --peak_oracle_weight 0.02 \
        --false_mass_weight 0.015 \
        --hit_mass_weight 0.003 \
        --oracle_bin_res 0.01 \
        --oracle_mz_tol 0.01 \
        --oracle_mz_sigma 0.003 \
        --oracle_low_w 0.50 \
        --oracle_mid_w 2.00 \
        --oracle_high_w 3.00 \
        --train_k3b \
        --train_formula_module \
        --train_frag_rep \
        --train_ce_flowfrag \
        --train_refiner \
        --train_render_gate

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R160 checkpoint已存在。"
fi

R160_COS="$(
    metric \
        "$OUT/08_R160/r160_best_val.csv"
)"

echo "R160 validation cosine: $R160_COS"


# =============================================================================
# R172D exact residual scorer
# =============================================================================

R172_PKL="$OUT/09_R172D/r170_regressor.pkl"

if [ ! -f "$R172_PKL" ]; then
    run_stage \
        "09_R172D" \
        python -u \
        "$DIAG/candidate_reranker.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R160_CKPT" \
        --out_dir "$OUT/09_R172D" \
        --seed "$R172_SEED" \
        --backend lightgbm \
        --max_train_rows "$R172_MAX_TRAIN_ROWS" \
        --neg_topk_per_batch "$R172_NEG_TOPK" \
        --neg_rand_per_batch "$R172_NEG_RANDOM" \
        --mz_tol 0.01 \
        --mz_sigma 0.003 \
        --target_bin_res 0.01 \
        --local_bin_res 0.01 \
        --eval_bin_res 0.01 \
        --residual_clip "$R172_RESIDUAL_CLIP" \
        --neg_residual "$R172_NEG_RESIDUAL" \
        --score_clip "$R172_SCORE_CLIP" \
        --low_w "$R172_LOW_W" \
        --mid_w "$R172_MID_W" \
        --high_w "$R172_HIGH_W" \
        --pos_weight "$R172_POS_W" \
        --pos_intensity_weight "$R172_POS_INTENSITY_W" \
        --neg_weight "$R172_NEG_W" \
        --neg_prob_weight "$R172_NEG_PROB_W" \
        --n_estimators "$R172_N_ESTIMATORS" \
        --gbdt_lr "$R172_GBDT_LR" \
        --num_leaves "$R172_NUM_LEAVES" \
        --max_depth "$R172_MAX_DEPTH" \
        --min_child_samples "$R172_MIN_CHILD" \
        --subsample "$R172_SUBSAMPLE" \
        --colsample_bytree "$R172_COLSAMPLE" \
        --reg_alpha "$R172_REG_ALPHA" \
        --reg_lambda "$R172_REG_LAMBDA" \
        --num_workers "$R172_WORKERS" \
        --max_extra_dims "$R172_EXTRA_DIMS" \
        --alpha_grid "$R172_ALPHA_GRID"

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R172D regressor已存在。"
fi

BEST_ALPHA="$(
    python "$ROOT/train/_impl/stage_metrics.py" \
        alpha \
        "$OUT/09_R172D/r170_alpha_val.csv"
)"

R172_COS="$(
    metric \
        "$OUT/09_R172D/r170_best_val.csv"
)"

echo "R172 selected alpha      : $BEST_ALPHA"
echo "R172 validation cosine  : $R172_COS"


# =============================================================================
# R184B sibling, stronger residual allocator
# =============================================================================

R184B_CKPT="$OUT/11_R184B/r184_allocator_best.pt"

if [ ! -f "$R184B_CKPT" ]; then
    run_stage \
        "11_R184B" \
        python -u \
        "$DIAG/spectrum_allocator.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$R160_CKPT" \
        --regressor_path "$R172_PKL" \
        --out_dir "$OUT/11_R184B" \
        --seed "$R172_SEED" \
        --epochs "$B_EPOCHS" \
        --lr "$B_LR" \
        --weight_decay "$B_WEIGHT_DECAY" \
        --grad_clip 5.0 \
        --hidden "$B_HIDDEN" \
        --layers "$B_LAYERS" \
        --dropout "$B_DROPOUT" \
        --score_clip "$B_SCORE_CLIP" \
        --alpha "$BEST_ALPHA" \
        --lgbm_score_clip "$B_LGBM_SCORE_CLIP" \
        --residual_scale "$B_RESIDUAL_SCALE" \
        --temperature "$B_TEMPERATURE" \
        --cos_weight "$B_COS_WEIGHT" \
        --jss_weight "$B_JSS_WEIGHT" \
        --target_ce_weight "$B_TARGET_CE_WEIGHT" \
        --pos_recall_weight "$B_POS_RECALL_WEIGHT" \
        --base_kl_weight "$B_BASE_KL_WEIGHT" \
        --residual_l2_weight "$B_RESIDUAL_L2_WEIGHT" \
        --low_w "$B_LOW_W" \
        --mid_w "$B_MID_W" \
        --high_w "$B_HIGH_W" \
        --mz_tol 0.01 \
        --mz_sigma 0.003 \
        --target_bin_res 0.01 \
        --local_bin_res 0.01 \
        --eval_bin_res 0.01 \
        --residual_clip 6.0 \
        --neg_residual 4.0 \
        --max_extra_dims "$R172_EXTRA_DIMS" \
        --max_train_batches 0

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] R184B allocator已存在。"
fi



echo
echo "============================================================"
echo "FINAL MAINLINE REFINEMENT COMPLETE"
echo "============================================================"
echo "Refined backbone checkpoint:"
echo "$R160_CKPT"
echo
echo "Candidate reranker:"
echo "$R172_PKL"
echo
echo "Final spectrum allocator:"
echo "$R184B_CKPT"
echo
echo "Test used for selection: False"
echo "============================================================"
