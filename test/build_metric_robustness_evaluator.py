from __future__ import annotations

from pathlib import Path


SOURCE = Path(
    "test/evaluate_chun_10ppm.py"
)

TARGET = Path(
    "test/evaluate_metric_robustness.py"
)

if not SOURCE.is_file():
    raise FileNotFoundError(
        SOURCE.resolve()
    )

text = SOURCE.read_text(
    encoding="utf-8"
)

helper_anchor = (
    "\n@torch.no_grad()\n"
    "def evaluate_split(\n"
)

if helper_anchor not in text:
    raise RuntimeError(
        "无法定位evaluate_split。"
    )

helper_code = r'''

def _metric_robustness_cosine_dense(
    true_dense: torch.Tensor,
    pred_dense: torch.Tensor,
) -> torch.Tensor:
    numerator = (
        true_dense
        * pred_dense
    ).sum(dim=1)

    true_norm = (
        true_dense
        .square()
        .sum(dim=1)
        .sqrt()
    )

    pred_norm = (
        pred_dense
        .square()
        .sum(dim=1)
        .sqrt()
    )

    denominator = (
        true_norm
        * pred_norm
    ).clamp_min(1.0e-12)

    return (
        numerator / denominator
    ).clamp(
        min=0.0,
        max=1.0,
    )


'''

text = text.replace(
    helper_anchor,
    helper_code + helper_anchor,
    1,
)

inference_anchor = (
    "        true_mzs, true_ints, true_batch = (\n"
)

if text.count(inference_anchor) != 1:
    raise RuntimeError(
        "无法唯一定位真实峰聚合位置。"
    )

metric_code = r'''        renderer_name = next(
            (
                name
                for name in (
                    "dense_by_round_bins_grad",
                    "dense_by_round_bins",
                )
                if callable(
                    getattr(
                        spectrum_allocator,
                        name,
                        None,
                    )
                )
            ),
            None,
        )

        if renderer_name is None:
            available = sorted(
                name
                for name in dir(
                    spectrum_allocator
                )
                if (
                    "dense" in name.lower()
                    or "bin" in name.lower()
                )
            )

            raise RuntimeError(
                "运行时spectrum_allocator中"
                "没有找到锁定dense renderer。"
                f"候选函数：{available}"
            )

        metric_renderer = getattr(
            spectrum_allocator,
            renderer_name,
        )

        batch_size = int(
            result["unique_id"].numel()
        )

        true_probability = (
            result["true_logprobs"]
            .exp()
            .float()
        )

        pred_probability = (
            output["new_logp"]
            .exp()
            .float()
        )

        true_batch_indices = (
            result["true_batch_idxs"]
            .long()
        )

        pred_batch_indices = (
            result["pred_batch_idxs"]
            .long()
        )

        mz_max = float(
            backbone.hparams.mz_max
        )

        locked_raw_cosine = (
            output["cos"]
            .detach()
            .float()
            .reshape(-1)
        )

        metric_raw_gpu = {}
        metric_sqrt_gpu = {}
        metric_recomputed_gpu = {}

        for (
            metric_label,
            metric_bin_res,
        ) in (
            ("0.01", 0.01),
            ("0.05", 0.05),
            ("0.10", 0.10),
        ):
            true_dense = metric_renderer(
                result["true_mzs"].float(),
                true_probability,
                true_batch_indices,
                batch_size=batch_size,
                mz_max=mz_max,
                bin_res=float(
                    metric_bin_res
                ),
            )

            pred_dense = metric_renderer(
                result["pred_mzs"].float(),
                pred_probability,
                pred_batch_indices,
                batch_size=batch_size,
                mz_max=mz_max,
                bin_res=float(
                    metric_bin_res
                ),
            )

            recomputed_raw = (
                _metric_robustness_cosine_dense(
                    true_dense,
                    pred_dense,
                )
            )

            sqrt_cosine = (
                _metric_robustness_cosine_dense(
                    true_dense
                    .clamp_min(0.0)
                    .sqrt(),
                    pred_dense
                    .clamp_min(0.0)
                    .sqrt(),
                )
            )

            metric_recomputed_gpu[
                metric_label
            ] = recomputed_raw

            metric_sqrt_gpu[
                metric_label
            ] = sqrt_cosine

            if metric_label == "0.01":
                metric_raw_gpu[
                    metric_label
                ] = locked_raw_cosine
            else:
                metric_raw_gpu[
                    metric_label
                ] = recomputed_raw

        raw_001_parity_abs = (
            metric_recomputed_gpu["0.01"]
            - locked_raw_cosine
        ).abs()

        metric_raw_cpu = {
            key: value.detach().cpu()
            for key, value
            in metric_raw_gpu.items()
        }

        metric_sqrt_cpu = {
            key: value.detach().cpu()
            for key, value
            in metric_sqrt_gpu.items()
        }

        recomputed_raw_001_cpu = (
            metric_recomputed_gpu["0.01"]
            .detach()
            .cpu()
        )

        raw_001_parity_abs_cpu = (
            raw_001_parity_abs
            .detach()
            .cpu()
        )

'''

text = text.replace(
    inference_anchor,
    metric_code + inference_anchor,
    1,
)

row_anchor = (
    '                    "chun_10ppm": chun,\n'
)

if text.count(row_anchor) != 1:
    raise RuntimeError(
        "无法唯一定位逐谱结果字段。"
    )

row_replacement = r'''                    "chun_10ppm": chun,
                    "cos_raw_0.01": float(
                        metric_raw_cpu[
                            "0.01"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.01": float(
                        metric_sqrt_cpu[
                            "0.01"
                        ][batch_index]
                    ),
                    "cos_raw_0.05": float(
                        metric_raw_cpu[
                            "0.05"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.05": float(
                        metric_sqrt_cpu[
                            "0.05"
                        ][batch_index]
                    ),
                    "cos_raw_0.10": float(
                        metric_raw_cpu[
                            "0.10"
                        ][batch_index]
                    ),
                    "cos_sqrt_0.10": float(
                        metric_sqrt_cpu[
                            "0.10"
                        ][batch_index]
                    ),
                    "cos_raw_0.01_recomputed_audit": float(
                        recomputed_raw_001_cpu[
                            batch_index
                        ]
                    ),
                    "raw_0.01_parity_abs": float(
                        raw_001_parity_abs_cpu[
                            batch_index
                        ]
                    ),
'''

text = text.replace(
    row_anchor,
    row_replacement,
    1,
)

summary_anchor = r'''                "chun_median": float(
                    current["chun_10ppm"].median()
                ),
'''

if text.count(summary_anchor) != 1:
    raise RuntimeError(
        "无法唯一定位summary字段。"
    )

summary_replacement = summary_anchor + r'''                "cos_raw_0.01": float(
                    current[
                        "cos_raw_0.01"
                    ].mean()
                ),
                "cos_sqrt_0.01": float(
                    current[
                        "cos_sqrt_0.01"
                    ].mean()
                ),
                "cos_raw_0.05": float(
                    current[
                        "cos_raw_0.05"
                    ].mean()
                ),
                "cos_sqrt_0.05": float(
                    current[
                        "cos_sqrt_0.05"
                    ].mean()
                ),
                "cos_raw_0.10": float(
                    current[
                        "cos_raw_0.10"
                    ].mean()
                ),
                "cos_sqrt_0.10": float(
                    current[
                        "cos_sqrt_0.10"
                    ].mean()
                ),
                "cos_raw_0.01_recomputed_audit": float(
                    current[
                        "cos_raw_0.01_recomputed_audit"
                    ].mean()
                ),
                "raw_0.01_parity_max_abs": float(
                    current[
                        "raw_0.01_parity_abs"
                    ].max()
                ),
'''

text = text.replace(
    summary_anchor,
    summary_replacement,
    1,
)

detail_anchor = r'''    detail = pd.DataFrame(rows)
    metrics = summarize(detail)
'''

if text.count(detail_anchor) != 1:
    raise RuntimeError(
        "无法唯一定位detail汇总位置。"
    )

detail_replacement = r'''    detail = pd.DataFrame(rows)

    if detail.empty:
        raise RuntimeError(
            f"{split}逐谱结果为空。"
        )

    maximum_renderer_parity_difference = float(
        detail[
            "raw_0.01_parity_abs"
        ].max()
    )

    mean_renderer_parity_difference = float(
        detail[
            "raw_0.01_parity_abs"
        ].mean()
    )

    print()
    print(
        f"[Metric robustness {split}] "
        f"renderer={renderer_name}"
    )
    print(
        f"[Metric robustness {split}] "
        "raw@0.01 parity: "
        f"max_abs="
        f"{maximum_renderer_parity_difference:.12e}, "
        f"mean_abs="
        f"{mean_renderer_parity_difference:.12e}"
    )

    if (
        maximum_renderer_parity_difference
        > 2.0e-6
    ):
        raise RuntimeError(
            f"{split} raw@0.01 renderer parity失败："
            f"max_abs="
            f"{maximum_renderer_parity_difference}"
        )

    metrics = summarize(detail)
'''

text = text.replace(
    detail_anchor,
    detail_replacement,
    1,
)

text = text.replace(
    'seed_dir / "chun_10ppm"',
    'seed_dir / "metric_robustness"',
    1,
)

text = text.replace(
    "LOCKED SEED42 CHUN-10PPM EVALUATION",
    "LOCKED METRIC ROBUSTNESS EVALUATION",
)

text = text.replace(
    "chun_10ppm_result.json",
    "metric_robustness_result.json",
)

compile(
    text,
    str(TARGET),
    "exec",
)

TARGET.write_text(
    text,
    encoding="utf-8",
)

print("已创建：", TARGET.resolve())
print("来源：", SOURCE.resolve())
print(
    "0.01 raw使用锁定output['cos']；"
    "重新渲染结果仅用于parity审计。"
)
