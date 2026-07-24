from __future__ import annotations

import ast
import re
from pathlib import Path


SOURCE = Path("test/evaluate_chun_10ppm.py")
TARGET = Path("test/benchmark_final_model.py")

source = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source)
lines = source.splitlines(keepends=True)


def target_names(target):
    if isinstance(target, ast.Name):
        return {target.id}

    if isinstance(target, (ast.Tuple, ast.List)):
        output = set()

        for item in target.elts:
            output.update(target_names(item))

        return output

    return set()


evaluate_function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "evaluate_split"
)

main_function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "main"
)

chun_assignments = []

for node in ast.walk(evaluate_function):
    if isinstance(node, ast.Assign):
        if any(
            "chun" in target_names(target)
            for target in node.targets
        ):
            chun_assignments.append(node)

if not chun_assignments:
    raise RuntimeError("没有找到CHUN赋值。")

test_assignment = None

for node in ast.walk(main_function):
    if not isinstance(node, ast.Assign):
        continue

    names = set()

    for target in node.targets:
        names.update(target_names(target))

    if "test_metrics" in names:
        test_assignment = node
        break

if test_assignment is None:
    raise RuntimeError("没有找到test评价调用。")

parse_return_marker = (
    "    return parser.parse_args()\n"
)

if parse_return_marker not in source:
    raise RuntimeError("无法定位parse_args返回。")

source = source.replace(
    parse_return_marker,
    '''    parser.add_argument(
        "--benchmark-repeat-id",
        type=int,
        default=1,
    )

''' + parse_return_marker,
    1,
)

tree = ast.parse(source)
lines = source.splitlines(keepends=True)

evaluate_function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "evaluate_split"
)

main_function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "main"
)

chun_assignments = []

for node in ast.walk(evaluate_function):
    if isinstance(node, ast.Assign):
        if any(
            "chun" in target_names(target)
            for target in node.targets
        ):
            chun_assignments.append(node)

test_assignment = None

for node in ast.walk(main_function):
    if not isinstance(node, ast.Assign):
        continue

    names = set()

    for target in node.targets:
        names.update(target_names(target))

    if "test_metrics" in names:
        test_assignment = node
        break

replacement_ranges = []

for assignment in chun_assignments:
    indent = re.match(
        r"\s*",
        lines[assignment.lineno - 1],
    ).group(0)

    replacement_ranges.append(
        (
            assignment.lineno - 1,
            assignment.end_lineno,
            f"{indent}chun = 0.0\n",
        )
    )

test_indent = re.match(
    r"\s*",
    lines[test_assignment.lineno - 1],
).group(0)

before_test = f'''{test_indent}if device.type == "cuda":
{test_indent}    torch.cuda.empty_cache()
{test_indent}    torch.cuda.reset_peak_memory_stats(device)
{test_indent}    torch.cuda.synchronize(device)

{test_indent}import time as _benchmark_time
{test_indent}_benchmark_start = _benchmark_time.perf_counter()

'''

after_test = f'''
{test_indent}if device.type == "cuda":
{test_indent}    torch.cuda.synchronize(device)

{test_indent}_benchmark_elapsed = (
{test_indent}    _benchmark_time.perf_counter()
{test_indent}    - _benchmark_start
{test_indent})

{test_indent}def _parameter_counts(module):
{test_indent}    if not hasattr(module, "parameters"):
{test_indent}        return {{"total": 0, "trainable": 0}}

{test_indent}    parameters = list(module.parameters())

{test_indent}    return {{
{test_indent}        "total": int(sum(
{test_indent}            parameter.numel()
{test_indent}            for parameter in parameters
{test_indent}        )),
{test_indent}        "trainable": int(sum(
{test_indent}            parameter.numel()
{test_indent}            for parameter in parameters
{test_indent}            if parameter.requires_grad
{test_indent}        )),
{test_indent}    }}

{test_indent}_test_count = int(len(test_detail))

{test_indent}_benchmark_result = {{
{test_indent}    "repeat_id": int(args.benchmark_repeat_id),
{test_indent}    "seed_dir": str(seed_dir),
{test_indent}    "spectrum_count": _test_count,
{test_indent}    "elapsed_seconds": float(_benchmark_elapsed),
{test_indent}    "seconds_per_spectrum": float(
{test_indent}        _benchmark_elapsed / max(_test_count, 1)
{test_indent}    ),
{test_indent}    "spectra_per_second": float(
{test_indent}        _test_count / max(_benchmark_elapsed, 1e-12)
{test_indent}    ),
{test_indent}    "backbone_parameters": _parameter_counts(
{test_indent}        backbone
{test_indent}    ),
{test_indent}    "allocator_parameters": _parameter_counts(
{test_indent}        allocator
{test_indent}    ),
{test_indent}    "artifact_bytes": {{
{test_indent}        "r160": int(backbone_path.stat().st_size),
{test_indent}        "r172d": int(reranker_path.stat().st_size),
{test_indent}        "r184b": int(allocator_path.stat().st_size),
{test_indent}    }},
{test_indent}    "device": str(device),
{test_indent}    "gpu_name": (
{test_indent}        torch.cuda.get_device_name(device)
{test_indent}        if device.type == "cuda"
{test_indent}        else None
{test_indent}    ),
{test_indent}    "peak_gpu_memory_allocated_bytes": (
{test_indent}        int(torch.cuda.max_memory_allocated(device))
{test_indent}        if device.type == "cuda"
{test_indent}        else 0
{test_indent}    ),
{test_indent}    "peak_gpu_memory_reserved_bytes": (
{test_indent}        int(torch.cuda.max_memory_reserved(device))
{test_indent}        if device.type == "cuda"
{test_indent}        else 0
{test_indent}    ),
{test_indent}    "chun_computation_disabled": True,
{test_indent}    "timing_scope": (
{test_indent}        "warm full-test inference after data/model preload"
{test_indent}    ),
{test_indent}}}

{test_indent}_benchmark_path = (
{test_indent}    output_dir
{test_indent}    / f"efficiency_repeat_{{args.benchmark_repeat_id}}.json"
{test_indent})

{test_indent}_benchmark_path.write_text(
{test_indent}    json.dumps(
{test_indent}        _benchmark_result,
{test_indent}        indent=2,
{test_indent}        ensure_ascii=False,
{test_indent}    )
{test_indent}    + "\\n",
{test_indent}    encoding="utf-8",
{test_indent})

{test_indent}print()
{test_indent}print("=" * 80)
{test_indent}print("EFFICIENCY BENCHMARK")
{test_indent}print("=" * 80)
{test_indent}print(json.dumps(
{test_indent}    _benchmark_result,
{test_indent}    indent=2,
{test_indent}    ensure_ascii=False,
{test_indent}))
'''

insertions = [
    (
        test_assignment.lineno - 1,
        before_test,
    ),
    (
        test_assignment.end_lineno,
        after_test,
    ),
]

for start, end, replacement in sorted(
    replacement_ranges,
    key=lambda item: item[0],
    reverse=True,
):
    lines[start:end] = [replacement]

for index, insertion in sorted(
    insertions,
    key=lambda item: item[0],
    reverse=True,
):
    lines[index:index] = [insertion]

generated = "".join(lines)

compile(
    generated,
    str(TARGET),
    "exec",
)

TARGET.write_text(
    generated,
    encoding="utf-8",
)

print("已创建：", TARGET.resolve())
print("CHUN匹配已在benchmark版本中关闭。")
