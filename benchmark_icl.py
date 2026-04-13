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

"""QCalEval Benchmark — In-context learning (ICL) evaluation.

Loads ICL benchmark from HuggingFace. Each entry has 3 QA pairs (Q3, Q5, Q6)
with <image> tags and demonstration examples. Images are resolved via image_ids
from the test split. Compatible with any OpenAI-compatible API endpoint.

Usage:
  python benchmark_icl.py --api-base https://api.openai.com/v1/chat/completions --model-id gpt-5.4 --api-key-env OPENAI_API_KEY --output results.json
  python benchmark_icl.py --api-base http://localhost:8000/v1/chat/completions --model-id my-model --api-key dummy --output results.json
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

FEWSHOT_Q_MAP = {
    0: "experimental_significance",
    1: "parameter_extraction",
    2: "calibration_diagnosis",
}


def encode_pil_image(pil_img):
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_image_index(ds_test):
    """Build image_id -> base64 content block lookup from test dataset."""
    index = {}
    for row in ds_test:
        for img_id, pil_img in zip(row["image_ids"], row["images"]):
            if img_id not in index and pil_img is not None:
                b64 = encode_pil_image(pil_img)
                index[img_id] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                }
    return index


def resolve_image_blocks(image_ids, image_index):
    """Resolve a list of image_ids to content blocks using the index."""
    blocks = []
    for img_id in image_ids:
        blocks.append(image_index.get(img_id))
    return blocks


def build_multimodal_content(question_text, image_blocks, image_offset):
    parts = question_text.split("<image>")
    content = []
    idx = image_offset
    if parts[0].strip():
        content.append({"type": "text", "text": parts[0].strip()})
    for part in parts[1:]:
        if idx < len(image_blocks) and image_blocks[idx] is not None:
            content.append(image_blocks[idx])
        idx += 1
        if part.strip():
            content.append({"type": "text", "text": part.strip()})
    return content, idx


def extract_content(response_data):
    msg = response_data["choices"][0]["message"]
    content = msg.get("content")
    if content is None:
        content = msg.get("reasoning") or ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
    return content.strip()


async def ask_icl_question(client, api_base, model_id, api_key,
                                content, qi, semaphore, max_tokens,
                                temperature, no_think=False):
    async with semaphore:
        for attempt in range(3):
            try:
                messages = [{"role": "user", "content": content}]
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
                    timeout=180.0,
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
                          image_blocks, semaphore, max_tokens, temperature,
                          idx, total, no_think=False):
    eid = row["source_id"]
    if not any(b for b in image_blocks):
        print(f"  [{idx+1}/{total}] {eid} SKIP: no images", flush=True)
        return {"id": eid, "error": "no images", "responses": {}}

    prompts = [row["q3_prompt"], row["q5_prompt"], row["q6_prompt"]]
    image_offset = 0
    tasks = []
    for qi in range(3):
        content, image_offset = build_multimodal_content(
            prompts[qi], image_blocks, image_offset
        )
        tasks.append(
            ask_icl_question(client, api_base, model_id, api_key,
                                content, qi, semaphore, max_tokens,
                                temperature, no_think)
        )

    results = await asyncio.gather(*tasks)
    responses = {}
    for qi, result in results:
        q_name = FEWSHOT_Q_MAP.get(qi)
        if q_name:
            responses[q_name] = result

    answered = sum(1 for v in responses.values() if v.get("answer"))
    print(f"  [{idx+1}/{total}] {eid} done ({answered}/3 questions)", flush=True)

    return {
        "id": eid,
        "experiment_type": row["experiment_type"],
        "error": None,
        "responses": responses,
    }


async def main():
    parser = argparse.ArgumentParser(description="QCalEval — In-context learning (ICL) evaluation")
    parser.add_argument("--api-base", type=str, required=True,
                        help="API base URL (OpenAI-compatible /v1/chat/completions)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Model ID for API request")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Env var for API key (default: OPENAI_API_KEY)")
    parser.add_argument("--api-key", type=str, help="API key directly")
    parser.add_argument("--dataset", type=str, default=DATASET_ID)
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

    # Load test split (for images) and ICL split
    print(f"Loading dataset: {args.dataset}")
    ds_test = load_dataset(args.dataset, split="test")
    from huggingface_hub import hf_hub_download
    icl_path = hf_hub_download(repo_id=args.dataset, filename="fewshot-00000-of-00001.parquet", repo_type="dataset")
    ds_icl = load_dataset("parquet", data_files=icl_path, split="train")

    # Build image_id -> base64 content block index
    print(f"Building image index from {len(ds_test)} test entries...")
    image_index = build_image_index(ds_test)
    print(f"  {len(image_index)} unique images indexed")

    # Resolve image blocks for each ICL entry
    entries = list(ds_icl)
    if args.filter_type:
        entries = [e for e in entries if args.filter_type in e["experiment_type"]]
    if args.limit:
        entries = entries[:args.limit]

    print(f"Resolving images for {len(entries)} ICL entries...")
    all_image_blocks = [resolve_image_blocks(e["image_ids"], image_index) for e in entries]

    total = len(entries)
    model_label = model_id
    print(f"\nQCalEval — In-context learning (ICL)")
    print(f"  Model: {model_label} ({model_id})")
    print(f"  API: {api_base}")
    print(f"  Entries: {total}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=240.0) as client:
        print("  Warmup: sending first entry...", flush=True)
        warmup = await benchmark_entry(
            client, api_base, model_id, api_key, entries[0], all_image_blocks[0],
            semaphore, args.max_tokens, args.temperature, 0, total,
            no_think=args.no_think)
        warmup_ok = sum(1 for v in warmup.get("responses", {}).values() if v.get("answer"))
        print(f"  Warmup done ({warmup_ok}/3 questions)\n", flush=True)

        tasks = [
            benchmark_entry(client, api_base, model_id, api_key, entries[i],
                           all_image_blocks[i], semaphore, args.max_tokens,
                           args.temperature, i, total, no_think=args.no_think)
            for i in range(total)
        ]
        results = await asyncio.gather(*tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "mode": "icl",
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

    answered = sum(1 for r in results for v in r.get("responses", {}).values() if v.get("answer"))
    print(f"\nDone.")
    print(f"  Entries: {len(results)}/{total}")
    print(f"  Questions answered: {answered}/{total * 3}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
