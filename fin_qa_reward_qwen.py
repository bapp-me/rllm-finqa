# Standard imports
import json
import re

import httpx
import openai
import os

from rllm.rewards.reward_types import RewardOutput

from .constants import (
    CORRECTNESS_PROMPT_PATH,
    MULTI_TABLE_CORRECTNESS_PROMPT_PATH,
)

with open(CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    CORRECTNESS_PROMPT = f.read()

with open(MULTI_TABLE_CORRECTNESS_PROMPT_PATH, encoding="utf-8") as f:
    MULTI_TABLE_CORRECTNESS_PROMPT = f.read()

JUDGE_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
JUDGE_API_KEY = os.getenv("api_key")
JUDGE_MODEL = "qwen3.5-flash"

custom_http_client = httpx.Client(
    http2=True,
    limits=httpx.Limits(max_connections=5000, max_keepalive_connections=2000),
    timeout=100.0,
    trust_env=True,
)

try:
    if not JUDGE_API_KEY:
        raise ValueError("Missing FINQA_REWARD_API_KEY (or WQ_API_KEY)")

    client_kwargs = {
        "api_key": JUDGE_API_KEY,
        "http_client": custom_http_client,
    }
    if JUDGE_API_BASE_URL:
        client_kwargs["base_url"] = JUDGE_API_BASE_URL

    JUDGE_CLIENT = openai.OpenAI(**client_kwargs)

except Exception as e:
    print(f"Warning: Failed to initialize global OpenAI client: {e}")
    JUDGE_CLIENT = None

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Weight configuration for multi-table scoring
CORRECTNESS_WEIGHTS = {
    "primary_data_score": 0.40,  # core correctness
    "derived_metrics_score": 0.40,  # core correctness
    "reasoning_score": 0.10,
    "consistency_score": 0.10,
}


def _call_judge(
    system_prompt: str,
    user_prompt: str,
    is_multi_table: bool = False,
) -> tuple[bool | float, dict, dict]:
    judge_stats = {
        "judge_request_ok": 0.0,
        "judge_client_unavailable": 0.0,
        "judge_timeout_error": 0.0,
        "judge_api_error": 0.0,
        "judge_parse_error": 0.0,
    }

    if JUDGE_CLIENT is None:
        judge_stats["judge_client_unavailable"] = 1.0
        print("[finqa_reward] Judge client unavailable; fallback reward path used.")
        return (False if not is_multi_table else 0.0), {}, judge_stats

    if not JUDGE_MODEL:
        judge_stats["judge_client_unavailable"] = 1.0
        print("[finqa_reward] FINQA_REWARD_MODEL is not set; fallback reward path used.")
        return (False if not is_multi_table else 0.0), {}, judge_stats

    if is_multi_table:
        user_prompt = (
            f"{user_prompt}\n\n"
            "Important: Return valid JSON only and include all required scoring fields."
        )

    if is_multi_table:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    request_kwargs = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "extra_body": {"enable_thinking": is_multi_table},
    }

    if is_multi_table:
        request_kwargs["max_tokens"] = 5000
    else:
        request_kwargs["max_tokens"] = 512

    try:
        response = JUDGE_CLIENT.chat.completions.create(**request_kwargs)
        message = response.choices[0].message if response and response.choices else None
        judge_output = message.content if message and message.content else ""

        if is_multi_table:
            parsed = {}
            try:
                parsed = json.loads(judge_output)
            except json.JSONDecodeError:
                json_match = _JSON_OBJECT_RE.search(judge_output)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        parsed = {}

            weighted_score = 0.0
            total_weight = 0.0
            for key, weight in CORRECTNESS_WEIGHTS.items():
                score = parsed.get(key)
                if isinstance(score, int | float):
                    normalized = float(score) / 5.0
                    weighted_score += normalized * weight
                    total_weight += weight

            overall = weighted_score / total_weight if total_weight > 0 else 0.0
            overall = max(0.0, min(1.0, overall))
            result = overall
        else:
            decision_text = judge_output.lower()
            decision = ("true" in decision_text) and ("false" not in decision_text)
            parsed = {}
            result = decision

        judge_stats["judge_request_ok"] = 1.0

        if is_multi_table and not parsed:
            judge_stats["judge_parse_error"] = 1.0
            print("[finqa_reward] Multi-table judge returned invalid JSON; reward fallback to 0 if no valid rubric fields.")

        return result, parsed, judge_stats

    except Exception as e:
        if isinstance(e, (openai.APITimeoutError, httpx.TimeoutException)):
            judge_stats["judge_timeout_error"] = 1.0
            print(f"[finqa_reward] Judge timeout: {type(e).__name__}: {e}")
        else:
            judge_stats["judge_api_error"] = 1.0
            print(f"[finqa_reward] Judge API/error: {type(e).__name__}: {e}")

        return (False if not is_multi_table else 0.0), {}, judge_stats


def _check_right_table_accessed(accessed_tables: list[str], expected_table_names: str | list[str]) -> float:
    """Return fraction of required tables that were accessed at least once."""
    if not accessed_tables or not expected_table_names:
        return 0.0

    normalized_access = {table.lower().strip() for table in accessed_tables if isinstance(table, str) and table.strip()}

    if isinstance(expected_table_names, list):
        expected = [name.lower().strip() for name in expected_table_names if isinstance(name, str) and name.strip()]
    else:
        expected = [expected_table_names.lower().strip()] if isinstance(expected_table_names, str) else []

    if not expected:
        return 0.0

    hits = sum(1 for name in expected if name in normalized_access)
    return hits / len(expected)


def _extract_final_answer(action: str, *, prefer_tail: bool = False) -> str:
    """Extract FINAL ANSWER section from model response."""
    # First try: handle code block format (```FINAL ANSWER: ... ```)
    code_match = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if code_match:
        return code_match.group(1).strip()

    # For long, multi-paragraph templates we often want everything after FINAL ANSWER:
    # In that case, skip the paragraph heuristic and fall back directly to the tail match.
    if not prefer_tail:
        # Second try: find FINAL ANSWER: and extract content until double newline
        match = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
        if match:
            return match.group(1).strip()

    # Third try: find FINAL ANSWER: and extract content until end of string
    match = _FINAL_ANSWER_TAIL_RE.search(action)
    if match:
        return match.group(1).strip()

    # Fallback: return entire action if no FINAL ANSWER found
    return action


def fin_qa_reward_function(task_info: dict, action: str) -> RewardOutput:
    """
    Calculate the reward for a financial question answering agent's action.

    Args:
        task_info: The task dictionary containing question, answer, and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value.
    """
    question = task_info.get("question")
    core_question = task_info.get("core_question") or question
    ground_truth = task_info.get("ground_truth")
    question_type = (task_info.get("question_type") or "").lower()
    curriculum_stage = str(task_info.get("curriculum_stage") or "single_table").lower()

    if not action or not question or not ground_truth:
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"correctness_reward": 0.0, "right_table_access_reward": 0.0},
        )

    is_multi_table = question_type.startswith("multi_table")

    # Build correctness input
    if is_multi_table:
        correctness_input = f"question : {core_question}\nmodel response : {action}\nlabel : {ground_truth}"
        system_prompt = MULTI_TABLE_CORRECTNESS_PROMPT
    else:
        final_answer = _extract_final_answer(action)
        correctness_input = f"question : {question}\nmodel response : {final_answer}\nlabel : {ground_truth}"
        system_prompt = CORRECTNESS_PROMPT

    result, rubric, judge_stats = _call_judge(
        system_prompt,
        correctness_input,
        is_multi_table=is_multi_table,
    )

    if is_multi_table:
        correctness_reward = float(result)
        is_correct = correctness_reward >= 0.9
    else:
        is_correct = bool(result)
        correctness_reward = 1.0 if is_correct else 0.0

    # ----- Efficiency Penalty Calculation -----
    # Count the number of '<tool_call>' tags in the action to represent the number of steps taken.
    num_steps = action.count("<tool_call>")
    step_penalty = num_steps * 0.01

    if is_multi_table:
        # For multi-table, apply penalty if the score is at least 0.85
        if correctness_reward >= 0.85:
            correctness_reward = max(0.0, correctness_reward - step_penalty)
    else:
        # For single-table, apply penalty if the score is at least 1.0
        if correctness_reward >= 1.0:
            correctness_reward = max(0.0, 1.0 - step_penalty)
    # ------------------------------------------

    # Check table access
    accessed_tables = task_info.get("accessed_tables", [])
    expected_table_names = task_info.get("table_name", "")
    right_table_access_reward = _check_right_table_accessed(accessed_tables, expected_table_names)

    # Build metadata
    metadata = {
        "correctness_reward": correctness_reward,
        "right_table_access_reward": right_table_access_reward,
        "is_single_table_sample": 0.0 if is_multi_table else 1.0,
        "is_multi_table_sample": 1.0 if is_multi_table else 0.0,
        "is_multi_table_medium_sample": 1.0 if curriculum_stage == "multi_table_medium" else 0.0,
        "is_multi_table_hard_sample": 1.0 if curriculum_stage == "multi_table_hard" else 0.0,
        "judge_request_ok": judge_stats["judge_request_ok"],
        "judge_client_unavailable": judge_stats["judge_client_unavailable"],
        "judge_timeout_error": judge_stats["judge_timeout_error"],
        "judge_api_error": judge_stats["judge_api_error"],
        "judge_parse_error": judge_stats["judge_parse_error"],
        # If judge request failed or parse failed, this flag helps dashboard filters.
        "judge_fallback_error": 1.0
        if any(
            judge_stats[k] > 0
            for k in ("judge_client_unavailable", "judge_timeout_error", "judge_api_error", "judge_parse_error")
        )
        else 0.0,
        "num_tool_calls": float(num_steps),
        "step_penalty_applied": float(step_penalty) if ((is_multi_table and is_correct) or (not is_multi_table and correctness_reward > 0.0)) else 0.0,
    }

    if is_multi_table:
        # Add all rubric scores to metadata
        for key in CORRECTNESS_WEIGHTS.keys():
            score = rubric.get(key)
            if isinstance(score, int | float):
                metadata[f"multi_table_{key}"] = float(score)
        metadata["multi_table_overall_score"] = correctness_reward

    return RewardOutput(
        reward=correctness_reward,
        is_correct=is_correct,
        metadata=metadata,
    )
