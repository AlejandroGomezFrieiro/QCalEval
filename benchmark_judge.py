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

"""QCalEval Benchmark Judge — scores model responses against ground truth.

Loads GT data from HuggingFace dataset. Combines programmatic scoring
(enums, booleans, binary) with LLM-based scoring (key points checklist).
Compatible with any OpenAI-compatible API endpoint for the LLM judge.

Usage:
  python benchmark_judge.py results.json --judge-api-base https://api.openai.com/v1/chat/completions --judge-model-id gpt-5.4
  python benchmark_judge.py results.json --judge-api-base http://localhost:8000/v1/chat/completions --judge-model-id my-model --judge-api-key dummy
"""
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATASET_ID = "nvidia/QCalEval"


Q_NAMES = [
    "technical_description",
    "experimental_conclusion",
    "experimental_significance",
    "fit_reliability",
    "parameter_extraction",
    "calibration_diagnosis",
]

# ─── Rubrics for GPT judge ───────────────────────────────────────────────────

CHECKLIST_PROMPT = """Check the model's answer against these key points.
For each key point, score: 1 if correctly addressed, 0.5 if partially addressed, 0 if missing or wrong.

Key points:
{key_points}

Respond with ONLY a JSON array of scores (one per key point):
[score1, score2, ...]"""



# ─── Programmatic scoring ────────────────────────────────────────────────────

def parse_json_answer(text):
    """Try to parse a JSON object or array from text."""
    if not text:
        return None
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find in code blocks
    m = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Find JSON object
    m = re.search(r'(\{[\s\S]*\})', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find JSON array
    m = re.search(r'(\[[\s\S]*\])', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def extract_classification(text):
    """Extract Classification: value from Q2 answer."""
    if not text:
        return None
    m = re.search(r'Classification:\s*\*?\*?(.+?)(?:\*\*)?$', text, re.IGNORECASE | re.MULTILINE)
    if m:
        val = m.group(1).strip().rstrip('.').replace('**', '').strip()
        # Normalize known values
        for known in ["Expected behavior", "Anomalous behavior", "Suboptimal parameters", "Apparatus issue"]:
            if known.lower() in val.lower():
                return known
        return val
    return None


def extract_assessment(text):
    """Extract Assessment: value from Q4 answer."""
    if not text:
        return None
    m = re.search(r'Assessment:\s*\*?\*?(.+?)(?:\*\*)?$', text, re.IGNORECASE | re.MULTILINE)
    if m:
        val = m.group(1).strip().rstrip('.').replace('**', '').strip()
        for known in ["Unreliable", "No fit", "Reliable"]:
            if known.lower() in val.lower():
                return known
        return val
    return None


def extract_yes_no(text):
    """Extract Yes/No from Q6 answer. Handles various formats."""
    if not text:
        return None
    clean = text.strip().replace('**', '').lower()
    # Direct start
    if clean.startswith('yes'):
        return 'Yes'
    if clean.startswith('no'):
        return 'No'
    # Search for "answer is Yes/No" or standalone Yes/No
    m = re.search(r'\b(?:answer\s+is\s+)?(?:\*\*)?(\byes\b|\bno\b)(?:\*\*)?', clean)
    if m:
        return 'Yes' if m.group(1) == 'yes' else 'No'
    return None


KNOWN_STATUSES = {
    "SUCCESS", "NO_SIGNAL", "BEATING", "NO_DETUNING", "SAMPLING_TOO_COARSE",
    "TOO_FEW_OSC", "TOO_MANY_OSC", "WINDOW_TOO_SHORT", "DAMPED", "FIT_POOR",
    "RANGE_TOO_NARROW", "AMP_TOO_HIGH", "NO_EXCITATION", "NO_RES_RESPONSE",
    "HIGH_POWER", "OPTIMAL_NOT_CENTERED", "LARGE_ERROR", "MODERATE_ERROR",
    "FIT_FAILED", "MULTIPLE_PEAKS", "INVERTED", "INCOMPLETE", "NO_TRANSITION",
    "NEGATIVE_OFFSET", "POSITIVE_OFFSET", "EVENT", "NO_EVENT", "NO_COHERENCE",
    "STABLE", "TELEGRAPHIC", "RANDOM_WALK", "ASYMMETRIC",
    "UNDERSAMPLED", "NOT_TUNABLE", "OFF_RESONANCE", "LOW_CONTRAST",
    "DETUNED", "NO_GATE", "MISCALIBRATED", "ABERRATED", "CORRECTED",
}


def extract_status(text):
    """Extract Status: or Classification: <value> from Q6 answer. Returns normalized uppercase string."""
    if not text:
        return None
    # Try "Status: VALUE" pattern first (most common, handles markdown bold)
    m = re.search(r'\bStatus:\s*\*{0,2}\s*([A-Z][A-Za-z_]+)', text)
    if m and m.group(1).upper() in KNOWN_STATUSES:
        return m.group(1).upper()
    # Try "Classification: VALUE" (charge tomography format)
    m = re.search(r'\bClassification:\s*\*{0,2}\s*([A-Z][A-Za-z_]+)', text)
    if m and m.group(1).upper() in KNOWN_STATUSES:
        return m.group(1).upper()
    # Fallback: match known status codes on first line
    first_line = text.strip().split("\n")[0].strip().upper()
    if first_line in KNOWN_STATUSES:
        return first_line
    # Fallback: find any known status in text using word boundaries
    # (longer strings first to avoid substring collisions)
    for status in sorted(KNOWN_STATUSES, key=len, reverse=True):
        if re.search(r'\b' + status + r'\b', text.upper()):
            return status
    return None


def extract_reason(text, field):
    """Extract reason text after Classification:/Assessment: line."""
    if not text:
        return ""
    m = re.search(rf'{field}:\s*.+?\n([\s\S]*)', text, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()
        # Remove "Reason:" prefix if present
        reason = re.sub(r'^Reason:\s*', '', reason, flags=re.IGNORECASE)
        return reason
    return text


def score_enum_match(model_val, gt_val):
    """Score exact enum match: 1.0 or 0.0."""
    if model_val is None or gt_val is None:
        return 0.0
    return 1.0 if str(model_val).lower().strip() == str(gt_val).lower().strip() else 0.0


def _score_q1_single(model_json, gt_json):
    """Score Q1 enum fields for a single image. Returns dict of field scores."""
    scores = {}
    if not isinstance(model_json, dict):
        return {"plot_type": 0.0, "x_scale": 0.0, "y_scale": 0.0}

    # plot_type
    scores["plot_type"] = score_enum_match(
        model_json.get("plot_type"), gt_json.get("plot_type")
    )
    # x_axis.scale — handle models returning string instead of dict
    model_x = model_json.get("x_axis", {})
    gt_x = gt_json.get("x_axis", {})
    scores["x_scale"] = score_enum_match(
        model_x.get("scale") if isinstance(model_x, dict) else None,
        gt_x.get("scale") if isinstance(gt_x, dict) else None
    )
    # y_axis.scale
    model_y = model_json.get("y_axis", {})
    gt_y = gt_json.get("y_axis", {})
    scores["y_scale"] = score_enum_match(
        model_y.get("scale") if isinstance(model_y, dict) else None,
        gt_y.get("scale") if isinstance(gt_y, dict) else None
    )
    return scores


def score_q1_programmatic(model_json, gt_json):
    """Score Q1 enum fields programmatically. Returns dict of field scores.

    For multi-image entries (JSON arrays), scores each image independently
    and averages across all images.
    """
    if model_json is None:
        return {"plot_type": 0.0, "x_scale": 0.0, "y_scale": 0.0}

    # Normalize to lists
    model_list = model_json if isinstance(model_json, list) else [model_json]
    gt_list = gt_json if isinstance(gt_json, list) else [gt_json]

    # Score each image; average over max(model, GT) count so extra
    # model images (hallucinations) are penalized with zero scores
    n = max(len(gt_list), len(model_list))
    all_scores = []
    for i in range(n):
        m = model_list[i] if i < len(model_list) else {}
        g = gt_list[i] if i < len(gt_list) and isinstance(gt_list[i], dict) else {}
        all_scores.append(_score_q1_single(m, g))

    # Average across all images
    fields = ["plot_type", "x_scale", "y_scale"]
    avg = {}
    for field in fields:
        vals = [s.get(field, 0.0) for s in all_scores]
        avg[field] = sum(vals) / len(vals) if vals else 0.0
    return avg


def score_q5_field(model_val, gt_val, spec):
    """Score one Q5 field programmatically. Returns 0.0-1.0."""
    # Unreliable handling
    if gt_val == "Unreliable":
        if model_val is None or (isinstance(model_val, str) and model_val.lower() in ("unreliable", "null", "none")):
            return 1.0
        return 0.0
    if gt_val is None:
        if model_val is None or (isinstance(model_val, str) and model_val.lower() in ("unreliable", "null", "none")):
            return 1.0
        return 0.0
    if model_val is None:
        return 0.0

    t = spec["type"]

    if t == "bool":
        # Accept bool, int 0/1, string "true"/"false"
        if isinstance(model_val, bool):
            mv = model_val
        elif isinstance(model_val, int) and model_val in (0, 1):
            mv = bool(model_val)
        elif isinstance(model_val, str) and model_val.lower() in ("true", "false"):
            mv = model_val.lower() == "true"
        else:
            return 0.0
        return 1.0 if mv == gt_val else 0.0

    if t == "enum":
        if isinstance(model_val, str) and isinstance(gt_val, str):
            return 1.0 if model_val.strip().lower() == gt_val.strip().lower() else 0.0
        return 0.0

    if t in ("int_count", "count_float"):
        # Handle arrays: use positional matching with absolute tolerance
        if isinstance(gt_val, list) or isinstance(model_val, list):
            gt_arr = gt_val if isinstance(gt_val, list) else [gt_val]
            model_arr = model_val if isinstance(model_val, list) else [model_val]
            if len(gt_arr) == 0 and len(model_arr) == 0:
                return 1.0
            if len(gt_arr) == 0 or len(model_arr) == 0:
                return 0.0
            n = min(len(model_arr), len(gt_arr))
            scores = []
            for i in range(n):
                try:
                    diff = abs(float(model_arr[i]) - float(gt_arr[i]))
                except (TypeError, ValueError):
                    scores.append(0.0)
                    continue
                if diff <= spec["tol_full"]:
                    scores.append(1.0)
                elif diff <= spec["tol_half"]:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            length_penalty = n / max(len(model_arr), len(gt_arr))
            return (sum(scores) / max(len(scores), 1)) * length_penalty
        try:
            diff = abs(float(model_val) - float(gt_val))
        except (TypeError, ValueError):
            return 0.0
        if diff <= spec["tol_full"]:
            return 1.0
        if diff <= spec["tol_half"]:
            return 0.5
        return 0.0

    if t == "pct":
        try:
            mv, gv = float(model_val), float(gt_val)
        except (TypeError, ValueError):
            return 0.0
        if gv == 0:
            return 1.0 if abs(mv) < 0.01 else 0.0
        ratio = abs(mv - gv) / abs(gv)
        if ratio <= spec["tol_full"]:
            return 1.0
        if ratio <= spec["tol_half"]:
            return 0.5
        return 0.0

    if t == "abs":
        try:
            diff = abs(float(model_val) - float(gt_val))
        except (TypeError, ValueError):
            return 0.0
        if diff <= spec["tol_full"]:
            return 1.0
        if diff <= spec["tol_half"]:
            return 0.5
        return 0.0

    if t == "coord_list":
        if not isinstance(model_val, list) or not isinstance(gt_val, list):
            return 0.0
        if len(model_val) != len(gt_val):
            return 0.0
        scores = []
        for mv, gv in zip(model_val, gt_val):
            try:
                diff = abs(float(mv) - float(gv))
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            if diff <= spec["tol_full"]:
                scores.append(1.0)
            elif diff <= spec["tol_half"]:
                scores.append(0.5)
            else:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    if t == "array_int_match":
        return score_array_f1(model_val, gt_val, spec["tol_full"])

    if t == "array_float_match":
        return score_array_float_match(model_val, gt_val, spec["tol_full"])

    return 0.0


def score_array_f1(model_arr, gt_arr, tolerance):
    """F1 score for variable-length int arrays with tolerance."""
    if not isinstance(model_arr, list) or not isinstance(gt_arr, list):
        return 0.0
    if len(gt_arr) == 0 and len(model_arr) == 0:
        return 1.0
    if len(gt_arr) == 0 or len(model_arr) == 0:
        return 0.0

    # Count hits: model positions that match a GT position within tolerance
    gt_matched = set()
    model_matched = set()
    for i, mv in enumerate(model_arr):
        for j, gv in enumerate(gt_arr):
            if j not in gt_matched:
                try:
                    if abs(float(mv) - float(gv)) <= tolerance:
                        model_matched.add(i)
                        gt_matched.add(j)
                        break
                except (TypeError, ValueError):
                    continue

    precision = len(model_matched) / max(len(model_arr), 1)
    recall = len(gt_matched) / max(len(gt_arr), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_array_float_match(model_arr, gt_arr, pct_tolerance):
    """Score variable-length float arrays. For matched elements, check pct tolerance."""
    if not isinstance(model_arr, list) or not isinstance(gt_arr, list):
        return 0.0
    if len(gt_arr) == 0 and len(model_arr) == 0:
        return 1.0
    if len(gt_arr) == 0:
        return 1.0  # no GT to compare against (e.g. empty jump_sizes_mV)
    if len(model_arr) == 0:
        return 0.0

    # Score element by element (positional match)
    n = min(len(model_arr), len(gt_arr))
    scores = []
    for i in range(n):
        try:
            mv, gv = float(model_arr[i]), float(gt_arr[i])
        except (TypeError, ValueError):
            scores.append(0.0)
            continue
        if gv == 0:
            scores.append(1.0 if abs(mv) < 0.01 else 0.0)
        else:
            ratio = abs(mv - gv) / abs(gv)
            scores.append(1.0 if ratio <= pct_tolerance else 0.0)

    # Penalize length mismatch
    length_penalty = n / max(len(model_arr), len(gt_arr))
    return (sum(scores) / max(len(scores), 1)) * length_penalty


def score_q5(model_json, gt_json, q5_scoring):
    """Score all Q5 fields programmatically using experiment config.

    Fields absent from GT JSON are skipped (not scored), so they don't
    inflate the average with free 1.0 points.
    """
    scores = {}
    if not gt_json or not q5_scoring:
        return scores
    for field, spec in q5_scoring.items():
        # Skip fields not present in GT (different from GT having null/Unreliable)
        if field not in gt_json:
            continue
        gt_val = gt_json[field]
        model_val = model_json.get(field) if isinstance(model_json, dict) else None
        scores[field] = score_q5_field(model_val, gt_val, spec)
    return scores


# ─── GPT Judge ───────────────────────────────────────────────────────────────

def build_judge_prompt(experiment_type, gt_q1, question_label, rubric,
                       gt_answer, model_answer):
    """Build the GPT judge prompt with Q1 context."""
    return f"""You are scoring a Vision-Language Model's response on a quantum experiment benchmark.

Experiment type: {experiment_type}

Ground truth plot description (Q1 — for context about the plot):
{gt_q1}

--- Scoring {question_label} ---

Ground truth answer:
{gt_answer}

Model's answer:
{model_answer}

{rubric}"""


def _sync_judge_post(api_base, model_id, api_key, prompt):
    """Synchronous HTTP POST for judge call (runs in thread pool)."""
    data = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        api_base,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode())


async def call_judge_structured(client, api_base, model_id, api_key, prompt, semaphore):
    """Call GPT judge and return parsed JSON response."""
    loop = asyncio.get_event_loop()
    async with semaphore:
        for attempt in range(5):
            try:
                status, body = await loop.run_in_executor(
                    None, _sync_judge_post, api_base, model_id, api_key, prompt
                )
                if status == 429:
                    await asyncio.sleep(10 * (2 ** attempt))
                    continue
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                # Parse JSON from response — try array first, then object
                m = re.search(r'\[[\s\S]*?\]', content)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                m = re.search(r'\{[\s\S]*\}', content)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
                raise ValueError(f"Judge returned no parseable JSON: {content[:300]}")
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError, OSError) as e:
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError("call_judge_structured: exhausted retries without return")




async def judge_entry(client, api_base, model_id, api_key, entry, model_result,
                      scoring_points, exp_config, type_to_family, semaphore, idx, total,
                      is_fewshot=False):
    """Score questions for one entry. Fewshot mode only scores Q3/Q5/Q6."""
    eid = entry["id"]
    exp_type = entry.get("experiment_type", "")
    convs = entry["conversations"]
    responses = model_result.get("responses", {})

    # Ground truth Q1 (always used as context)
    gt_q1 = convs[1]["value"] if len(convs) > 1 else ""

    # Get scoring points for this entry
    sp = scoring_points.get(eid, {})
    q1_key_points = sp.get("q1_key_points", [])
    q3_key_points = sp.get("q3_key_points", {})

    scores = {}

    # ── Q1: Technical Description (skip in fewshot mode) ──
    if not is_fewshot:
        gt_q1_json = parse_json_answer(gt_q1)
        model_q1 = responses.get("technical_description", {}).get("answer", "")
        model_q1_json = parse_json_answer(model_q1)

        # Programmatic: enums (plot_type, scales)
        q1_prog = score_q1_programmatic(model_q1_json, gt_q1_json)
        q1_prog_score = sum(q1_prog.values()) / max(len(q1_prog), 1)  # 0-1

        # GPT: key points checklist
        if q1_key_points:
            kp_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(q1_key_points))
            checklist = CHECKLIST_PROMPT.format(key_points=kp_text)
            prompt = build_judge_prompt(exp_type, gt_q1, "Q1 (Technical Description)",
                                        checklist, gt_q1, model_q1)
            kp_resp = await call_judge_structured(client, api_base, model_id, api_key, prompt, semaphore)
            if isinstance(kp_resp, list):
                expected = len(q1_key_points)
                def _safe_float(v):
                    try:
                        return max(0.0, min(1.0, float(v)))
                    except (TypeError, ValueError):
                        return 0.0
                kp_hits = [_safe_float(v) for v in kp_resp[:expected]]
                kp_hits += [0.0] * (expected - len(kp_hits))
            else:
                raise ValueError(f"Q1 judge bad response for {eid}: {kp_resp}")
            kp_ratio = sum(kp_hits) / max(len(q1_key_points), 1)
        else:
            kp_hits = []
            kp_ratio = 0.0

        q1_final = (q1_prog_score * 100 * 0.5) + (kp_ratio * 100 * 0.5)
        scores["Q1"] = {
            "score": round(q1_final, 1),
            "programmatic": {k: v for k, v in q1_prog.items()},
            "key_points_hit": kp_hits,
            "key_points_total": len(q1_key_points),
        }

    # ── Q2: Experimental Conclusion (skip in fewshot mode) ──
    if not is_fewshot:
        gt_q2 = convs[3]["value"] if len(convs) > 3 else ""
        model_q2 = responses.get("experimental_conclusion", {}).get("answer", "")

        gt_class = extract_classification(gt_q2)
        model_class = extract_classification(model_q2)
        q2_correct = gt_class and model_class and gt_class.lower() == model_class.lower()

        scores["Q2"] = {
            "score": 100.0 if q2_correct else 0.0,
            "correct": bool(q2_correct),
            "gt_classification": gt_class,
            "model_classification": model_class,
        }

    # ── Q3: Experimental Significance ──
    gt_q3 = convs[5]["value"] if len(convs) > 5 else ""
    model_q3 = responses.get("experimental_significance", {}).get("answer", "")

    kp = q3_key_points or {}
    # Build key points list from either format
    if "guidance_1" in kp:
        q3_kp_list = [kp["guidance_1"], kp["guidance_2"], kp["guidance_3"]]
    elif "behavior" in kp:
        q3_kp_list = [kp["behavior"], kp["validity"], kp["next_step"]]
    else:
        q3_kp_list = []

    if q3_kp_list:
        kp_text = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(q3_kp_list))
        checklist = CHECKLIST_PROMPT.format(key_points=kp_text)
        prompt = build_judge_prompt(exp_type, gt_q1, "Q3 (Experimental Significance)",
                                    checklist, gt_q3, model_q3)
        kp_resp = await call_judge_structured(client, api_base, model_id, api_key, prompt, semaphore)
        if isinstance(kp_resp, list):
            # Truncate/pad to expected length, clamp values to [0, 1]
            expected = len(q3_kp_list)
            def _safe_float_q3(v):
                try:
                    return max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    return 0.0
            q3_hits = [_safe_float_q3(v) for v in kp_resp[:expected]]
            q3_hits += [0.0] * (expected - len(q3_hits))
        else:
            raise ValueError(f"Q3 judge bad response for {eid}: {kp_resp}")
        q3_ratio = sum(q3_hits) / max(len(q3_kp_list), 1)
    else:
        q3_hits = []
        q3_ratio = 0.0

    q3_final = q3_ratio * 100
    scores["Q3"] = {
        "score": round(q3_final, 1),
        "key_points_hit": q3_hits,
        "key_points_total": len(q3_kp_list),
    }

    # ── Q4: Fit Reliability (skip in fewshot mode) ──
    if not is_fewshot:
        gt_q4 = convs[7]["value"] if len(convs) > 7 else ""
        model_q4 = responses.get("fit_reliability", {}).get("answer", "")

        gt_assess = extract_assessment(gt_q4)
        model_assess = extract_assessment(model_q4)
        q4_correct = gt_assess and model_assess and gt_assess.lower() == model_assess.lower()

        scores["Q4"] = {
            "score": 100.0 if q4_correct else 0.0,
            "correct": bool(q4_correct),
            "gt_assessment": gt_assess,
            "model_assessment": model_assess,
        }

    # ── Q5: Parameter Extraction ──
    gt_q5 = convs[9]["value"] if len(convs) > 9 else ""
    model_q5 = responses.get("parameter_extraction", {}).get("answer", "")

    gt_q5_json = parse_json_answer(gt_q5)
    model_q5_json = parse_json_answer(model_q5)

    family = type_to_family.get(exp_type, "")
    q5_scoring_spec = exp_config.get(family, {}).get("q5_scoring", {})
    q5_field_scores = score_q5(model_q5_json, gt_q5_json, q5_scoring_spec)
    q5_final = (sum(q5_field_scores.values()) / max(len(q5_field_scores), 1)) * 100 if q5_field_scores else 0

    scores["Q5"] = {
        "score": round(q5_final, 1),
        "fields": q5_field_scores,
    }

    # ── Q6: Calibration Diagnosis (Status Match) ──
    gt_q6 = convs[-1]["value"] if len(convs) >= 2 else ""
    model_q6 = responses.get("calibration_diagnosis", {}).get("answer", "")

    gt_status = extract_status(gt_q6)
    model_status = extract_status(model_q6)
    # Aliases: accept either status as correct for ambiguous failure modes
    Q6_ALIASES = {
        "TOO_MANY_OSC": {"SAMPLING_TOO_COARSE"},
        "SAMPLING_TOO_COARSE": {"TOO_MANY_OSC"},
    }
    q6_correct = (
        gt_status and model_status and
        (gt_status == model_status or model_status in Q6_ALIASES.get(gt_status, set()))
    )

    scores["Q6"] = {
        "score": 100.0 if q6_correct else 0.0,
        "correct": bool(q6_correct),
        "gt_status": gt_status,
        "model_status": model_status,
    }

    # ── Overall ──
    avg_score = sum(s["score"] for s in scores.values()) / len(scores)
    scores["overall"] = round(avg_score, 1)

    q_strs = " ".join(f"Q{i+1}={scores[f'Q{i+1}']['score']:.0f}" for i in range(6) if f"Q{i+1}" in scores)
    print(f"  [{idx+1}/{total}] {eid}: {q_strs} avg={avg_score:.1f}", flush=True)

    return {
        "id": eid,
        "experiment_type": exp_type,
        "scores": scores,
    }


def hf_row_to_entry(row):
    """Convert a HuggingFace dataset row to the entry format judge_entry expects."""
    conversations = []
    for qi in range(1, 7):
        conversations.append({"from": "human", "value": row[f"q{qi}_prompt"]})
        conversations.append({"from": "gpt", "value": row[f"q{qi}_answer"]})
    return {
        "id": row["id"],
        "experiment_type": row["experiment_type"],
        "conversations": conversations,
    }


def hf_row_to_scoring_points(row):
    """Convert HF row to scoring points entry."""
    q3_kp = row.get("q3_key_points", [])
    q3_dict = {}
    for i, kp in enumerate(q3_kp):
        q3_dict[f"guidance_{i+1}"] = kp
    return {
        "entry_id": row["id"],
        "q1_key_points": row.get("q1_key_points", []),
        "q3_key_points": q3_dict,
    }


async def main():
    parser = argparse.ArgumentParser(description="QCalEval Benchmark Judge")
    parser.add_argument("results_file", help="Path to model results JSON")
    parser.add_argument("--judge-api-base", type=str, required=True,
                        help="Judge LLM API base URL (OpenAI-compatible)")
    parser.add_argument("--judge-model-id", type=str, required=True,
                        help="Judge LLM model ID")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY",
                        help="Env var for judge API key (default: OPENAI_API_KEY)")
    parser.add_argument("--judge-api-key", type=str, help="Judge API key directly")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=DATASET_ID,
                        help=f"HuggingFace dataset ID (default: {DATASET_ID})")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_base = args.judge_api_base
    model_id = args.judge_model_id
    api_key = args.judge_api_key or os.environ.get(args.judge_api_key_env, "")
    if not api_key:
        print(f"ERROR: Set {args.judge_api_key_env} or use --judge-api-key")
        sys.exit(1)

    # Load model results
    with open(args.results_file) as f:
        model_data = json.load(f)
    model_results = model_data["results"]

    # Load GT from HuggingFace
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="test")

    entries_by_id = {}
    scoring_points = {}
    for row in ds:
        entry = hf_row_to_entry(row)
        entries_by_id[entry["id"]] = entry
        scoring_points[row["id"]] = hf_row_to_scoring_points(row)
    print(f"  Loaded {len(entries_by_id)} GT entries")

    # Load experiment config from HF repo
    config_path = hf_hub_download(repo_id=args.dataset, filename="experiment_config.json", repo_type="dataset")
    with open(config_path) as f:
        exp_config = json.load(f)

    type_to_family = {}
    for family, cfg in exp_config.items():
        for exp_type_key in cfg.get("q6_status_mapping", {}):
            type_to_family[exp_type_key] = family
    print(f"  Experiment config: {len(exp_config)} families, {len(type_to_family)} experiment types")

    # Match model results to GT entries
    pairs = []
    for mr in model_results:
        eid = mr["id"]
        if eid in entries_by_id and mr.get("responses"):
            pairs.append((entries_by_id[eid], mr))

    total = len(pairs)
    print(f"\nQCalEval Benchmark Judge")
    print(f"  Model results: {args.results_file}")
    print(f"  Mode: {model_data.get('mode', 'unknown')}")
    print(f"  Tested model: {model_data.get('model', 'unknown')}")
    print(f"  Judge: {model_id}")
    print(f"  Entries to judge: {total}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)
    is_fewshot = model_data.get("mode", "") in ("fewshot", "icl")

    client = None
    tasks = [
        judge_entry(client, api_base, model_id, api_key, entry, mr,
                   scoring_points, exp_config, type_to_family, semaphore, idx, total,
                   is_fewshot=is_fewshot)
        for idx, (entry, mr) in enumerate(pairs)
    ]
    results = await asyncio.gather(*tasks)

    # ── Aggregate scores ──
    def _median(vals):
        """Proper median: average two middle values for even-length lists."""
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    scored_qs = [f"Q{i+1}" for i in range(6)]
    if is_fewshot:
        scored_qs = ["Q3", "Q5", "Q6"]

    q_scores = {ql: [] for ql in scored_qs}
    overall_scores = []

    for r in results:
        for ql in scored_qs:
            if ql in r["scores"]:
                q_scores[ql].append(r["scores"][ql]["score"])
        overall_scores.append(r["scores"]["overall"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_data.get('model', 'unknown')} ({model_data.get('mode', '?')})")
    print(f"{'='*60}")
    print(f"{'Question':<30} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for ql in scored_qs:
        s = q_scores[ql]
        if s:
            mean = sum(s) / len(s)
            median = _median(s)
            print(f"{ql:<30} {mean:>8.1f} {median:>8.1f} {min(s):>8.1f} {max(s):>8.1f}")

    if overall_scores:
        mean = sum(overall_scores) / len(overall_scores)
        median = _median(overall_scores)
        print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'Overall':<30} {mean:>8.1f} {median:>8.1f} {min(overall_scores):>8.1f} {max(overall_scores):>8.1f}")

    # Classification accuracy (only for scored questions)
    q2_correct = 0
    q4_correct = 0
    q6_correct = sum(1 for r in results if r["scores"].get("Q6", {}).get("correct"))
    print(f"\nClassification Accuracy:")
    if not is_fewshot:
        q2_correct = sum(1 for r in results if r["scores"].get("Q2", {}).get("correct"))
        q4_correct = sum(1 for r in results if r["scores"].get("Q4", {}).get("correct"))
        print(f"  Q2 (Classification): {q2_correct}/{total} ({q2_correct/total*100:.1f}%)")
        print(f"  Q4 (Assessment):     {q4_correct}/{total} ({q4_correct/total*100:.1f}%)")
    print(f"  Q6 (Diagnosis):      {q6_correct}/{total} ({q6_correct/total*100:.1f}%)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_path = args.output
    else:
        results_name = Path(args.results_file).stem
        output_path = f"judged_{results_name}_{timestamp}.json"

    output = {
        "judge_model": model_id,
        "judge_model_id": model_id,
        "tested_model": model_data.get("model", "unknown"),
        "tested_model_id": model_data.get("model_id", "unknown"),
        "mode": model_data.get("mode", "unknown"),
        "timestamp": timestamp,
        "total_entries": total,
        "aggregate": {
            ql: {
                "mean": round(sum(q_scores[ql]) / len(q_scores[ql]), 1) if q_scores[ql] else 0,
                "median": round(_median(q_scores[ql]), 1) if q_scores[ql] else 0,
            }
            for ql in scored_qs
        },
        "overall_mean": round(sum(overall_scores) / len(overall_scores), 1) if overall_scores else 0,
        "classification_accuracy": {
            **({"Q2": round(q2_correct / total * 100, 1)} if not is_fewshot else {}),
            **({"Q4": round(q4_correct / total * 100, 1)} if not is_fewshot else {}),
            "Q6": round(q6_correct / total * 100, 1) if total else 0,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
