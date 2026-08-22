import json
import os
import random
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")

SYSTEM_PROMPT = """
You are a senior software architecture reviewer.

Your task is to generate a clear, evidence-based reasoning process for a
classification result of a Java class.

You will receive:

1. The source code of the class.
2. A Metrics Expert report.
3. A Semantic Audit report.
4. A Responsibility Expert report.
5. The final classification label.

Label definitions:

- 1: The class is a God Class. It contains excessive responsibilities,
  low cohesion, high complexity, broad coupling, or mixes unrelated concerns.
- 0: The class is not a God Class. It remains reasonably focused and cohesive.
- -1: The result is uncertain, invalid, or the available evidence is
  insufficient or contradictory.

Your reasoning must:

1. Identify the primary purpose of the class.
2. Summarize the most relevant evidence from the Metrics Expert report.
3. Summarize the main design findings from the Semantic Audit report.
4. Analyze the method-level responsibility and cosine-distance findings from
   the Responsibility Expert report.
5. Explain how the three reports jointly support or contradict the final label.
6. Mention important evidence that weakens the conclusion, if applicable.
7. Avoid inventing facts that are not present in the code or reports.
8. Synthesize the evidence instead of simply repeating the reports.
9. Preserve the provided label and do not modify it.

Return only a valid JSON object with exactly this field:

{
  "reasoning_process": "A concise but detailed evidence-based explanation."
}

Write the reasoning in professional English.
"""


def call_deepseek(
    client: OpenAI,
    code: str,
    metrics_report: dict,
    semantic_report: dict,
    responsibility_report: dict,
    label: int,
    max_retries: int = 2,
):
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
Analyze the following Java class and explain why the provided final label is
appropriate.

## Source Code
{code}

## Metrics Expert Report
{json.dumps(metrics_report, ensure_ascii=False)}

## Semantic Audit Report
{json.dumps(semantic_report, ensure_ascii=False)}

## Responsibility Expert Report
{json.dumps(responsibility_report, ensure_ascii=False)}

## Final Label
{label}
""",
                    },
                ],
                temperature=0,
                timeout=60,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("model returned empty content")

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"raw_content": content, "error": "not valid JSON"}

        except Exception as exc:
            tqdm.write(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt >= max_retries:
                raise

            delay = min(60.0, 2**attempt) + random.random()
            tqdm.write(f"Retrying in {delay:.1f}s")
            time.sleep(delay)

    raise AssertionError("retry loop ended unexpectedly")


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    if not DEEPSEEK_URL:
        raise RuntimeError("DEEPSEEK_URL is missing")

    print("Reading data files...")
    metrics = load_jsonl("./sub_agent_result/train_metrics_result.jsonl")
    semantics = load_jsonl("./sub_agent_result/train_semantic_result.jsonl")
    responsibilities = load_jsonl(
        "./sub_agent_result/train_responsibility_result.jsonl"
    )
    codes = load_jsonl("./dataset/train_data.jsonl")

    lengths = {
        "metrics": len(metrics),
        "semantics": len(semantics),
        "responsibilities": len(responsibilities),
        "codes": len(codes),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Input file lengths do not match: {lengths}")

    total = len(codes)
    print(f"Total: {total} items to process")

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_URL,
        max_retries=0,
    )

    failed_indices = []
    success_count = 0

    # Write each result immediately so completed work survives an interruption.
    with open("./results.jsonl", "w", encoding="utf-8") as output_file:
        for index in tqdm(range(total), desc="Progress", unit="items"):
            try:
                result = call_deepseek(
                    client=client,
                    code=codes[index].get("code", ""),
                    metrics_report=metrics[index],
                    semantic_report=semantics[index],
                    responsibility_report=responsibilities[index],
                    label=codes[index].get("label", 0),
                    max_retries=2,
                )
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
                success_count += 1
            except Exception as exc:
                failed_indices.append(index)
                tqdm.write(
                    f"Item {index} failed: {type(exc).__name__}: {str(exc)[:300]}"
                )

    if failed_indices:
        with open("./failed_indices.json", "w", encoding="utf-8") as file:
            json.dump(failed_indices, file, ensure_ascii=False, indent=2)

    print(f"Done! Success: {success_count}, Failed: {len(failed_indices)}")


if __name__ == "__main__":
    run()
