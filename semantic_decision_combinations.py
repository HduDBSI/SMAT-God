#!/usr/bin/env python3
"""Evaluate every decision-model and Semantic Audit result combination."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from decision_agent_separated import (
        SeparatedDecisionManager,
        SplitPaths,
        load_aligned_data,
    )
except ImportError:
    # The separated implementation may be deployed as decision_agent.py.
    from decision_agent import (  # type: ignore[no-redef]
        SeparatedDecisionManager,
        SplitPaths,
        load_aligned_data,
    )

from decision_result_utils import save_json_atomic


DEFAULT_DECISION_MODELS = (
    ("DeepSeek-R1-Distill-Llama-8B", "/model/bd_zjx/model/DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen2.5-Coder-7B-Instruct", "/model/bd_zjx/model/Qwen2.5-Coder-7B-Instruct"),
)

DEFAULT_SEMANTIC_RESULTS = (
    Path("sub_agent_result/test_semantic_result.jsonl"),
    Path("sub_agent_result/no_fine_tuning_test_semantic_result.jsonl"),
    Path("sub_agent_result/CodeLlama-7b-Instruct-hf_semantic_result.jsonl"),
    Path("sub_agent_result/deepseek-coder-7b-instruct-v1.5_semantic_result.jsonl"),
)


@dataclass(frozen=True)
class DecisionModelSpec:
    name: str
    path: str
    adapter: Path | None = None


def _named_value(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} must use NAME=PATH format, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"{option} must contain a non-empty NAME and PATH")
    return name.strip(), path.strip()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "unnamed"


def _semantic_name(path: Path) -> str:
    suffix = "_semantic_result"
    return path.stem[:-len(suffix)] if path.stem.endswith(suffix) else path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-model",
        action="append",
        metavar="NAME=PATH",
        help="Decision model; repeat for multiple models (defaults to DeepSeek-R1 and Qwen2.5-Coder)",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=PATH",
        help="Optional LoRA adapter associated with a --decision-model NAME",
    )
    parser.add_argument(
        "--semantic-result",
        action="append",
        type=Path,
        help="Row-aligned Semantic Audit JSONL; repeat for multiple results",
    )
    parser.add_argument("--data-path", type=Path, default=Path("dataset/test_data.jsonl"))
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("sub_agent_result/test_metrics_result.jsonl"),
    )
    parser.add_argument(
        "--responsibility-path",
        type=Path,
        default=Path("sub_agent_result/test_responsibility_result.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/semantic_decision_combinations"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--no-responsibility", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed combination and continue with the remaining combinations",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all aligned input files without loading any model",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def build_model_specs(args: argparse.Namespace) -> list[DecisionModelSpec]:
    try:
        model_pairs = (
            [_named_value(value, "--decision-model") for value in args.decision_model]
            if args.decision_model
            else list(DEFAULT_DECISION_MODELS)
        )
        adapter_map = dict(
            _named_value(value, "--adapter") for value in (args.adapter or [])
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model_names = [name for name, _ in model_pairs]
    if len(model_names) != len(set(model_names)):
        raise SystemExit("--decision-model names must be unique")
    unknown_adapters = sorted(set(adapter_map) - set(model_names))
    if unknown_adapters:
        raise SystemExit(
            "Adapters reference undefined decision models: " + ", ".join(unknown_adapters)
        )
    return [
        DecisionModelSpec(name, path, Path(adapter_map[name]) if name in adapter_map else None)
        for name, path in model_pairs
    ]


def build_paths(args: argparse.Namespace, semantic_path: Path) -> SplitPaths:
    return SplitPaths(
        data=args.data_path,
        metrics=args.metrics_path,
        semantic=semantic_path,
        responsibility=args.responsibility_path,
        reasoning=None,
    )


def validate_inputs(args: argparse.Namespace, semantic_paths: list[Path]) -> int:
    expected_count = None
    for semantic_path in semantic_paths:
        rows = load_aligned_data(
            build_paths(args, semantic_path),
            use_metrics=not args.no_metrics,
            use_semantics=True,
            use_responsibility=not args.no_responsibility,
            include_reasoning=False,
        )
        if expected_count is None:
            expected_count = len(rows)
        elif len(rows) != expected_count:
            raise ValueError(
                f"Semantic result row counts differ: expected {expected_count}, "
                f"but {semantic_path} produced {len(rows)}"
            )
        print(f"validated: {semantic_path} ({len(rows)} records)")
    return expected_count or 0


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or not isinstance(value.get("combinations"), dict):
        raise ValueError(f"Invalid experiment summary: {path}")
    return value["combinations"]


def save_summary(path: Path, combinations: dict[str, dict[str, Any]]) -> None:
    save_json_atomic({"combinations": combinations}, path)


def release_manager(manager: SeparatedDecisionManager) -> None:
    manager.model = None
    manager.tokenizer = None
    del manager
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    args = parse_args()
    model_specs = build_model_specs(args)
    semantic_paths = args.semantic_result or list(DEFAULT_SEMANTIC_RESULTS)
    record_count = validate_inputs(args, semantic_paths)
    print(
        f"input validation complete: {len(model_specs)} decision models x "
        f"{len(semantic_paths)} semantic results = "
        f"{len(model_specs) * len(semantic_paths)} combinations, {record_count} records each"
    )
    if args.validate_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    combinations = load_summary(summary_path) if not args.no_resume else {}

    for spec in model_specs:
        manager = SeparatedDecisionManager(
            spec.path,
            device_map={"": args.gpu},
            use_semantics=True,
            use_metrics=not args.no_metrics,
            use_responsibility=not args.no_responsibility,
        )
        try:
            if spec.adapter is not None:
                manager.load_adapter(str(spec.adapter))

            for semantic_path in semantic_paths:
                semantic_name = _semantic_name(semantic_path)
                combination_id = f"{spec.name}__{semantic_name}"
                output_path = args.output_dir / f"{_slug(combination_id)}_results.json"
                print(f"evaluating: {spec.name} x {semantic_name}")
                try:
                    metrics = manager.test_local(
                        build_paths(args, semantic_path),
                        batch_size=args.batch_size,
                        output_path=str(output_path),
                        resume=not args.no_resume,
                    )
                    combinations[combination_id] = {
                        "decision_model": spec.name,
                        "model_path": spec.path,
                        "adapter_path": str(spec.adapter) if spec.adapter else None,
                        "semantic_result": semantic_name,
                        "semantic_path": str(semantic_path),
                        "result_path": str(output_path),
                        "status": "completed",
                        **metrics,
                    }
                except Exception as exc:
                    combinations[combination_id] = {
                        "decision_model": spec.name,
                        "model_path": spec.path,
                        "adapter_path": str(spec.adapter) if spec.adapter else None,
                        "semantic_result": semantic_name,
                        "semantic_path": str(semantic_path),
                        "result_path": str(output_path),
                        "status": "failed",
                        "error": str(exc),
                    }
                    save_summary(summary_path, combinations)
                    if not args.continue_on_error:
                        raise
                    print(f"failed: {combination_id}: {exc}")
                    continue
                save_summary(summary_path, combinations)
                print(f"completed: {combination_id}: {json.dumps(metrics, ensure_ascii=False)}")
        finally:
            release_manager(manager)

    print(f"summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
