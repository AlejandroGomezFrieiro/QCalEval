#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QCalEval Benchmark — Zero-shot evaluation.

Loads benchmark from HuggingFace, sends image(s) + each question
independently (6 separate requests per entry). Compatible with any
OpenAI-compatible API endpoint.

Usage:
  python benchmark_zeroshot.py --api-base https://api.openai.com/v1/chat/completions --model-id gpt-5.4 --api-key-env OPENAI_API_KEY --output results.json
  python benchmark_zeroshot.py --api-base http://localhost:8000/v1/chat/completions --model-id my-model --api-key dummy --output results.json
"""
import argparse
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from io import BytesIO

import httpx
from datasets import load_dataset

DATASET_ID = "nvidia/QCalEval"

Q_NAMES = [
    "technical_description",
    "experimental_conclusion",
    "experimental_significance",
    "fit_reliability",
    "parameter_extraction",
    "calibration_diagnosis",
]

Q_PROMPT_KEYS = [f"q{i}_prompt" for i in range(1, 7)]


def encode_pil_image(pil_img):
    """Encode a PIL image to base64 PNG string."""
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_image_contents(row):
    """Build image content blocks from HF dataset row (PIL images)."""
    contents = []
    for img in row["images"]:
        if img is not None:
            b64 = encode_pil_image(img)
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })
    return contents


def extract_content(response_data):
    """Extract text from API response, handling reasoning models."""
    msg = response_data["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        content = msg.get("reasoning") or ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
    return content.strip()


async def ask_single_question(client, api_base, model_id, api_key,
                               image_contents, question, qi, semaphore,
                               max_tokens, temperature, no_think=False):
    """Send a single question with image(s) and return the response."""
    async with semaphore:
        for attempt in range(3):
            try:
                messages = [{
                    "role": "user",
                    "content": image_contents + [{"type": "text", "text": question}]
                }]
                extra_params = {}
                if no_think:
                    extra_params["chat_template_kwargs"] = {"enable_thinking": False}
                resp = await client.post(
                    api_base,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        **extra_params,
                    },
                    timeout=300.0,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                answer = extract_content(resp.json())
                return qi, {"answer": answer, "error": None}
            except Exception as e:
                if attempt == 2:
                    return qi, {"answer": None, "error": str(e)}
                await asyncio.sleep(2 ** attempt)

        return qi, {"answer": None, "error": "max_retries"}


async def benchmark_entry(client, api_base, model_id, api_key, row,
                          image_contents, semaphore, max_tokens, temperature,
                          idx, total, no_think=False):
    """Run zero-shot benchmark on all 6 questions for one entry."""
    eid = row["id"]

    if not image_contents:
        print(f"  [{idx+1}/{total}] {eid} SKIP: no images", flush=True)
        return {"id": eid, "error": "no images", "responses": {}}

    questions = [row[k] for k in Q_PROMPT_KEYS]

    tasks = [
        ask_single_question(client, api_base, model_id, api_key,
                           image_contents, questions[qi], qi, semaphore,
                           max_tokens, temperature, no_think=no_think)
        for qi in range(6)
    ]
    results = await asyncio.gather(*tasks)

    responses = {}
    for qi, result in results:
        responses[Q_NAMES[qi]] = result

    answered = sum(1 for v in responses.values() if v.get("answer"))
    print(f"  [{idx+1}/{total}] {eid} done ({answered}/6 questions)", flush=True)

    return {
        "id": eid,
        "experiment_type": row["experiment_type"],
        "error": None,
        "responses": responses,
    }


async def main():
    parser = argparse.ArgumentParser(description="QCalEval — Zero-shot evaluation")
    parser.add_argument("--api-base", type=str, required=True,
                        help="API base URL (OpenAI-compatible /v1/chat/completions)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model ID for API request")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Env var for API key (default: OPENAI_API_KEY)")
    parser.add_argument("--api-key", type=str, help="API key directly")
    parser.add_argument("--dataset", type=str, default=DATASET_ID,
                        help=f"HuggingFace dataset ID (default: {DATASET_ID})")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter-type", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking (for reasoning models like Qwen3.5)")
    args = parser.parse_args()

    api_base = args.api_base
    model_id = args.model_id
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"ERROR: Set {args.api_key_env} or use --api-key")
        sys.exit(1)

    # Load benchmark from HuggingFace
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="test")

    if args.filter_type:
        ds = ds.filter(lambda x: args.filter_type in x["experiment_type"])
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Pre-encode images (avoid re-encoding per question)
    print(f"Encoding {len(ds)} entries' images...")
    all_image_contents = [get_image_contents(row) for row in ds]

    total = len(ds)
    model_label = model_id
    print(f"\nQCalEval — Zero-shot")
    print(f"  Model: {model_label} ({model_id})")
    print(f"  API: {api_base}")
    print(f"  Entries: {total}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=240.0) as client:
        # Warmup
        print("  Warmup: sending first entry...")
        warmup = await benchmark_entry(
            client, api_base, model_id, api_key, ds[0], all_image_contents[0],
            semaphore, args.max_tokens, args.temperature, 0, total,
            no_think=args.no_think)
        warmup_ok = sum(1 for v in warmup.get("responses", {}).values() if v.get("answer"))
        print(f"  Warmup done ({warmup_ok}/6 questions)\n")

        tasks = [
            benchmark_entry(client, api_base, model_id, api_key, ds[idx],
                           all_image_contents[idx], semaphore, args.max_tokens,
                           args.temperature, idx, total, no_think=args.no_think)
            for idx in range(total)
        ]
        results = await asyncio.gather(*tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "mode": "zeroshot",
        "model": model_label,
        "model_id": model_id,
        "api_base": api_base,
        "dataset": args.dataset,
        "timestamp": timestamp,
        "total_entries": total,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    answered = sum(1 for r in results for v in r["responses"].values() if v.get("answer"))
    total_qs = sum(len(r["responses"]) for r in results)
    print(f"\nDone.")
    print(f"  Entries: {len(results)}/{total}")
    print(f"  Questions answered: {answered}/{total_qs}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
