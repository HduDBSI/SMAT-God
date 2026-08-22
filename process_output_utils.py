#!/usr/bin/env python3
"""Result parsing, persistence, repair, and evaluation for decision agents."""

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def normalize_binary_label(value: Any) -> int | None:
    try:
        if isinstance(value, str) and not value.strip():
            return None
        label = int(float(value))
    except (TypeError, ValueError):
        return None
    return label if label in (0, 1) else None


def normalize_model_output(raw_output: Any) -> str:
    if raw_output is None:
        return ""
    text = str(raw_output).replace("Ċ", "\n").replace("Ġ", " ")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"```(?:json)?", "```", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))[-20:]):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def extract_prediction(raw_output: Any) -> tuple[int | None, str]:
    """Return (judge_label, reasoning_process) from a model response."""
    if isinstance(raw_output, dict):
        objects = [raw_output]
        text = json.dumps(raw_output, ensure_ascii=False)
    else:
        text = normalize_model_output(raw_output)
        objects = list(_json_objects(text))

    for value in objects:
        label = normalize_binary_label(value.get("judge_label"))
        if label is not None:
            reasoning = value.get("reasoning_process", "")
            return label, str(reasoning) if reasoning is not None else ""


    boxed_start = text.rfind(r"\boxed")
    if boxed_start >= 0:
        boxed_tail = text[boxed_start:boxed_start + 1500]
        keyed = re.findall(
            r"(?:judge(?:s)?|range)(?:\\?[_ ]|\s)*"
            r"(?:label|value)[^01]{0,80}([01])",
            boxed_tail,
            flags=re.IGNORECASE,
        )
        if keyed:
            return int(keyed[-1]), ""

    boxed = list(re.finditer(r"\\boxed\s*\{(.*?)\}", text, flags=re.DOTALL))
    if boxed:
        content = boxed[-1].group(1)
        keyed = re.findall(
            r"(?:judge\s*[_ ]?label|judges\s*[_ ]?label|range\s*[_ ]?value)"
            r"[^01]{0,80}([01])",
            content,
            flags=re.IGNORECASE,
        )
        if keyed:
            return int(keyed[-1]), ""
        if re.fullmatch(r"\s*[01]\s*", content):
            return int(content.strip()), ""



    patterns = (
        r'["\']?judge[_ ]label["\']?\s*[:=]\s*["\']?([01])',
        r'final(?:\s+judge)?\s*label\s*(?:is|=|:)\s*["\']?([01])',
        r'verdict\s*(?:is|=|:)\s*["\']?([01])',
        r'final\s+(?:json\s+)?object[^\n]{0,120}?'
        r'(?:judge\s*[_ ]?label|judges\s*[_ ]?label|range\s*[_ ]?value)'
        r'[^01]{0,80}([01])',
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text, flags=re.IGNORECASE))
    if matches:
        match = max(matches, key=lambda item: item.start())
        return int(match.group(1)), ""
    return None, ""


def compute_binary_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = total = 0
    for row in rows:
        label = normalize_binary_label(row.get("label"))
        if label is None:
            continue
        total += 1
        prediction = normalize_binary_label(row.get("pred_label"))
        if prediction is None:
            invalid += 1
        elif label == prediction == 1:
            tp += 1
        elif label == prediction == 0:
            tn += 1
        elif label == 0:
            fp += 1
        else:
            fn += 1

    valid = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / valid if valid else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "samples": total,
        "valid_predictions": valid,
        "invalid_predictions": invalid,
        "coverage": valid / total if total else 0.0,
    }


def save_json_atomic(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def load_result_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and isinstance(data.get("results"), dict):
        return data["results"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Result file must contain a JSON object: {path}")


class ResultWriter:
    """Incrementally persist predictions and support index-based resume."""

    def __init__(self, path: str | Path | None, resume: bool = True):
        self.path = Path(path) if path else None
        self.results = load_result_rows(self.path) if self.path and resume else {}

    @property
    def completed_indices(self) -> set[int]:
        return {int(index) for index in self.results}

    def add(self, index: int, label: Any, raw_output: Any, error: str | None = None) -> None:
        cleaned = normalize_model_output(raw_output)
        pred_label, reasoning = extract_prediction(cleaned)
        row = {
            "output": cleaned,
            "reasoning_process": reasoning,
            "label": normalize_binary_label(label),
            "pred_label": pred_label,
        }
        if error:
            row["error"] = error
        self.results[str(index)] = row
        self.save()

    def save(self) -> None:
        if self.path:
            payload = {
                "metrics": compute_binary_metrics(self.results.values()),
                "results": self.results,
            }
            save_json_atomic(payload, self.path)

    def summary(self) -> dict[str, Any]:
        return compute_binary_metrics(self.results.values())


def repair_result_file(path: str | Path) -> dict[str, Any]:
    """Reparse invalid predictions in an existing result file and rewrite it."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    results = payload.get("results", payload)
    repaired = 0
    for row in results.values():
        if normalize_binary_label(row.get("pred_label")) is not None:
            continue
        label, reasoning = extract_prediction(row.get("output", ""))
        if label is not None:
            row["pred_label"] = label
            row["reasoning_process"] = reasoning
            repaired += 1
    output = {"metrics": compute_binary_metrics(results.values()), "results": results}
    save_json_atomic(output, path)
    return {"repaired": repaired, **output["metrics"]}
