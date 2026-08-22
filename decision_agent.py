#!/usr/bin/env python3
"""Decision agent for row-aligned main datasets and separate sub-agent reports."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import gc
from tqdm import tqdm
from dotenv import load_dotenv
from process_output_utils import ResultWriter, normalize_binary_label, save_json_atomic, compute_binary_metrics


SYSTEM_PROMPT = """You are the Lead Architect and Final Arbiter in a code-smell detection system.
Determine whether the supplied Java class is a God Class by synthesizing only the enabled expert reports.
A God Class centralizes excessive or unrelated responsibilities and commonly shows poor cohesion, excessive complexity, or broad coupling.
Return only a valid JSON object with exactly these fields:
{"reasoning_process": "concise evidence-based reasoning", "judge_label": 0}
judge_label must be 1 for God Class or 0 for not God Class.
"""


@dataclass(frozen=True)
class SplitPaths:
    data: Path
    metrics: Path
    semantic: Path
    responsibility: Path
    reasoning: Path | None=None

def default_split_paths(split: str, root: str | Path = ".") -> SplitPaths:
    root = Path(root)
    return SplitPaths(
        data=root / "dataset" / f"{split}_data.jsonl",
        metrics=root / "sub_agent_result" / f"{split}_metrics_result.jsonl",
        semantic=root / "sub_agent_result" / f"{split}_semantic_result.jsonl",
        responsibility=root / "sub_agent_result" / f"{split}_responsibility_result.jsonl",
        reasoning=(root / "dataset" / f"reasoning_result.jsonl" if split=='train' else None)
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def load_aligned_data(
    paths: SplitPaths,
    *,
    use_metrics: bool = True,
    use_semantics: bool = True,
    use_responsibility: bool = True,
    include_reasoning=True
) -> list[dict[str, Any]]:
    """Merge files by row index after strictly validating their lengths."""
    data = load_jsonl(paths.data)
    sources = {}
    if use_metrics:
        sources["Metrics"] = load_jsonl(paths.metrics)
    if use_semantics:
        sources["Semantic Audit"] = load_jsonl(paths.semantic)
    if use_responsibility:
        sources["Responsibility Gap"] = load_jsonl(paths.responsibility)

    for field, records in sources.items():
        if len(records) != len(data):
            raise ValueError(
                f"Row count mismatch: {paths.data} has {len(data)} records, "
                f"but {field} has {len(records)}"
            )
    if include_reasoning:
        reasoning_records = load_jsonl(paths.reasoning)

        if len(reasoning_records) != len(data):
            raise ValueError(
                f"Row count mismatch: {paths.data} has {len(data)} records, "
                f"but reasoning results have {len(reasoning_records)}"
            )
    merged = []
    for index, record in enumerate(data):
        row = dict(record)

        # 合并三个子 Agent 报告
        for field, records in sources.items():
            row[field] = records[index]
        if include_reasoning:
            row["reasoning_process"] = reasoning_records[index]["reasoning_process"]
        merged.append(row)
    return merged


def evaluate_local_model_combinations(
    model_ids: list[str] | None = None,
    semantic_test_paths: list[str | Path] | None = None,
    *,
    data_path: str | Path = "./dataset/test_data.jsonl",
    metrics_path: str | Path = "./sub_agent_result/test_metrics_result.jsonl",
    responsibility_path: str | Path = "./sub_agent_result/test_responsibility_result.jsonl",
    batch_size: int = 8,
    output_dir: str | Path = "./output/semantic_decision_combinations",
    device_map: Any = None,
    use_metrics: bool = True,
    use_responsibility: bool = True,
    resume: bool = True,
) -> dict[str, dict[str, Any]]:
    """Evaluate base decision models against every aligned Semantic Result.

    No decision LoRA adapter is loaded. Each model is initialized once and then
    reused across all semantic-result files; ``test_local`` persists individual
    predictions so an interrupted combination can resume by row index.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    model_ids = model_ids or [
        "/model/bd_zjx/model/Llama-3.1-8B-Instruct",
        "/model/bd_zjx/model/Qwen2.5-Math-7B-Instruct",
        "/model/bd_zjx/model/DeepSeek-R1-Distill-Llama-8B",
    ]
    semantic_test_paths = semantic_test_paths or [
        "./sub_agent_result/test_semantic_result.jsonl",
        "./sub_agent_result/CodeLlama-7b-Instruct-hf_semantic_result.jsonl",
        "./sub_agent_result/deepseek-coder-7b-instruct-v1.5_semantic_result.jsonl",
    ]
    semantic_test_paths = [Path(path) for path in semantic_test_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary: dict[str, dict[str, Any]] = {}
    if resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict) and isinstance(payload.get("combinations"), dict):
            summary = payload["combinations"]

    def model_name(model_id: str) -> str:
        return Path(model_id.rstrip("/")).name or model_id.replace("/", "_")

    def semantic_name(path: Path) -> str:
        suffix = "_semantic_result"
        return path.stem[:-len(suffix)] if path.stem.endswith(suffix) else path.stem
    def existing_metrics(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid result file: {path}")
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            return metrics
        results = payload.get("results", payload)
        if not isinstance(results, dict):
            raise ValueError(f"Result rows must be a JSON object: {path}")
        return compute_binary_metrics(results.values())


    for model_id in model_ids:
        manager = Decision_Agent(
            model_id,
            device_map={"": 0} if device_map is None else device_map,
            use_semantics=True,
            use_metrics=use_metrics,
            use_responsibility=use_responsibility,
        )
        try:
            for semantic_path in semantic_test_paths:
                model_label = model_name(model_id)
                semantic_label = semantic_name(semantic_path)
                combination = f"{model_label}__{semantic_label}"
                result_path = output_dir / f"{combination}_results.json"
                if resume and result_path.exists():
                    metrics = existing_metrics(result_path)
                    summary[combination] = {
                        "model_id": model_id,
                        "semantic_path": str(semantic_path),
                        "result_path": str(result_path),
                        "adapter_path": None,
                        **metrics,
                    }
                    save_json_atomic({"combinations": summary}, summary_path)
                    print(
                        f"{combination}: existing metrics "
                        f"{json.dumps(metrics, ensure_ascii=False)}"
                    )
                    continue
                paths = SplitPaths(
                    data=Path(data_path),
                    metrics=Path(metrics_path),
                    semantic=semantic_path,
                    responsibility=Path(responsibility_path),
                )
                metrics = manager.test_local(
                    paths,
                    batch_size=batch_size,
                    output_path=str(result_path),
                    resume=resume,
                )
                summary[combination] = {
                    "model_id": model_id,
                    "semantic_path": str(semantic_path),
                    "result_path": str(result_path),
                    "adapter_path": None,
                    **metrics,
                }
                save_json_atomic({"combinations": summary}, summary_path)
                print(f"{combination}: {json.dumps(metrics, ensure_ascii=False)}")
        finally:
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
    return summary



class Decision_Agent:
    def __init__(
        self,
        model_id: str,
        device_map: Any = None,
        *,
        use_semantics: bool = True,
        use_metrics: bool = True,
        use_responsibility: bool = True,
        initialize_model: bool = False,
    ):
        self.model_id = model_id
        self.device_map = {"": 0} if device_map is None else device_map
        self.use_semantics = use_semantics
        self.use_metrics = use_metrics
        self.use_responsibility = use_responsibility
        self.model = None
        self.tokenizer = None
        if not any((use_semantics, use_metrics, use_responsibility)):
            raise ValueError("At least one sub-agent report must be enabled")
        if initialize_model:
            self._init_tokenizer()
            self._init_base_model()

    def _init_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.clean_up_tokenization_spaces = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_base_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        self._init_tokenizer()
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = True

    @staticmethod
    def _format_report(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value

    def build_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        sections = [f"## Source Code\n\n{row['code']}"]
        if self.use_metrics:
            sections.append("## Metrics Expert Report\n\n" + self._format_report(row["Metrics"]))
        if self.use_semantics:
            sections.append("## Semantic Audit Report\n\n" + self._format_report(row["Semantic Audit"]))
        if self.use_responsibility:
            sections.append(
                "## Responsibility Expert Report\n\n" + self._format_report(row["Responsibility Gap"])
            )
        sections.append("Explain the evidence and return the required JSON object.")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _training_messages(self, row: dict[str, Any]) -> list[dict[str, str]]:
        label = normalize_binary_label(row.get("label"))
        if label is None:
            raise ValueError(f"Invalid training label: {row.get('label')!r}")
        reasoning = row.get("reasoning_process", row.get("Reasoning Process", ""))
        answer = {"reasoning_process": str(reasoning), "judge_label": label}
        return self.build_messages(row) + [
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}
        ]

    def train(
        self,
        paths: SplitPaths,
        output_dir: str = "./manager_agent_separated",
        *,
        epochs: int = 3,
    ) -> None:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

        rows = load_aligned_data(
            paths,
            use_metrics=self.use_metrics,
            use_semantics=self.use_semantics,
            use_responsibility=self.use_responsibility,
        )
        self._init_base_model()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ))

        def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
            messages = self._training_messages(row)
            full = self.tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            full_ids = self.tokenizer(full, truncation=True, max_length=4096)["input_ids"]
            prompt_ids = self.tokenizer(prompt, truncation=True, max_length=4096)["input_ids"]
            return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}

        dataset = Dataset.from_list([tokenize(row) for row in tqdm(rows, desc="Tokenizing training set")])
        args = TrainingArguments(
            output_dir="./manager_separated_checkpoints",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=10,
            num_train_epochs=epochs,
            learning_rate=3e-5,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            save_total_limit=1,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, padding=True),
        )
        trainer.train()
        self.model.save_pretrained(output_dir)

    def load_adapter(self, no_adapter: bool, adapter_path: str) -> None:
        from peft import PeftModel

        self._init_base_model()
        if not no_adapter:
            print('load Adapter')
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def judge_batch(self, rows: list[dict[str, Any]], max_new_tokens: int = 1024) -> list[str]:
        import torch

        if self.model is None:
            self._init_base_model()
        messages = [self.build_messages(row) for row in rows]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def test_local(
        self,
        paths: SplitPaths,
        *,
        batch_size: int = 8,
        output_path: str | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        rows = load_aligned_data(
            paths,
            use_metrics=self.use_metrics,
            use_semantics=self.use_semantics,
            use_responsibility=self.use_responsibility,
            include_reasoning=False,
        )
        writer = ResultWriter(output_path, resume=resume)
        pending = [index for index in range(len(rows)) if index not in writer.completed_indices]
        for start in tqdm(range(0, len(pending), batch_size), desc="Local decision evaluation"):
            indices = pending[start:start + batch_size]
            outputs = self.judge_batch([rows[index] for index in indices])
            for index, output in zip(indices, outputs):
                writer.add(index, rows[index]["label"], output)
        return writer.summary()

    def judge_api(self, row: dict[str, Any], client: Any, api_model: str, **options: Any) -> str:
        payload = {
            "model": api_model,
            "messages": self.build_messages(row),
            "temperature": options.pop("temperature", 0.1),
            **options,
        }
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""

    def test_api(
        self,
        paths: SplitPaths,
        client: Any,
        api_model: str,
        *,
        concurrency: int = 8,
        output_path: str | None = None,
        resume: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        rows = load_aligned_data(
            paths,
            use_metrics=self.use_metrics,
            use_semantics=self.use_semantics,
            use_responsibility=self.use_responsibility,
            include_reasoning=False
        )
        writer = ResultWriter(output_path, resume=resume)
        pending = [index for index in range(len(rows)) if index not in writer.completed_indices]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self.judge_api, rows[index], client, api_model, **options): index
                for index in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="API decision evaluation"):
                index = futures[future]
                try:
                    output = future.result()
                    writer.add(index, rows[index]["label"], output)
                except Exception as exc:
                    tqdm.write(f"record {index} failed: {exc}")
        return writer.summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "local", "api"))
    parser.add_argument("--split", choices=("train", "test"), default=None)
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default="./manager_agent_separated")
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base decision model without loading a LoRA adapter",
    )
    parser.add_argument("--output", default="./output/separated_decision_results.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--semantic-path",
        help="Override the row-aligned Semantic Audit JSONL file",
    )
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--no-responsibility", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_URL"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    split = args.split or ("train" if args.mode == "train" else "test")
    paths = default_split_paths(split, args.root)
    if args.semantic_path:
        paths = replace(paths, semantic=Path(args.semantic_path))
    
    manager = Decision_Agent(
        args.model,
        use_metrics=not args.no_metrics,
        use_semantics=not args.no_semantic,
        use_responsibility=not args.no_responsibility,
    )
    if args.mode == "train":
        manager.train(paths, args.adapter)
    elif args.mode == "local":
        manager.load_adapter(args.no_adapter, args.adapter)
        print(manager.test_local(
            paths,
            batch_size=args.batch_size,
            output_path=args.output,
            resume=not args.no_resume,
        ))
    else:
        from openai import OpenAI
        if not args.api_key or not args.base_url:
            raise RuntimeError("API mode requires --api-key and --base-url")
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        print(manager.test_api(
            paths,
            client,
            args.model,
            concurrency=args.concurrency,
            output_path=args.output,
            resume=not args.no_resume,
        ))


if __name__ == "__main__":
    main()

    # summary = evaluate_local_model_combinations(
    #     batch_size=8,
    #     device_map={"": 0},
    #     output_dir="./output/semantic_decision_combinations",
    # )

