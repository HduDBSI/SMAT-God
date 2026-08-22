import asyncio
import json
import os
import random
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")

CONCURRENCY = 10
REQUEST_TIMEOUT = 90
MAX_RETRIES = 2

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


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_response(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"raw_content": content, "error": "not valid JSON"}


async def call_deepseek(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    index: int,
    code: str,
    metrics_report: dict,
    semantic_report: dict,
    responsibility_report: dict,
    label: int,
) -> tuple[int, dict | None, Exception | None]:
    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
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
                    timeout=REQUEST_TIMEOUT,
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("model returned empty content")

                return index, parse_response(content), None

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                tqdm.write(
                    f"Item {index}, attempt {attempt + 1}/{MAX_RETRIES + 1} "
                    f"failed: {type(exc).__name__}: {str(exc)[:300]}"
                )
                if attempt >= MAX_RETRIES:
                    return index, None, exc

                delay = min(60.0, 2**attempt) + random.random()
                await asyncio.sleep(delay)

    raise AssertionError("retry loop ended unexpectedly")


async def run():
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
    print(f"Total: {total} items, concurrency: {CONCURRENCY}")

    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_URL,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        asyncio.create_task(
            call_deepseek(
                client=client,
                semaphore=semaphore,
                index=index,
                code=codes[index].get("code", ""),
                metrics_report=metrics[index],
                semantic_report=semantics[index],
                responsibility_report=responsibilities[index],
                label=codes[index].get("label", 0),
            )
        )
        for index in range(total)
    ]

    results: list[dict | None] = [None] * total
    failed_indices = []

    try:
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=total,
            desc="Progress",
            unit="items",
        ):
            index, result, error = await future
            if error is None:
                results[index] = result
            else:
                failed_indices.append(index)
    finally:
        await client.close()

    with open("./results_async.jsonl", "w", encoding="utf-8") as output_file:
        for result in results:
            if result is not None:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    if failed_indices:
        with open("./failed_indices_async.json", "w", encoding="utf-8") as file:
            json.dump(sorted(failed_indices), file, ensure_ascii=False, indent=2)

    print(
        f"Done! Success: {total - len(failed_indices)}, "
        f"Failed: {len(failed_indices)}"
    )


if __name__ == "__main__":
    asyncio.run(run())
