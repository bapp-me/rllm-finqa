# Standard imports
import json
import re
import time

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

# Judge API configuration: supports both Doubao (Responses API) and local vLLM (Chat Completions API)
# Set FINQA_JUDGE_API_TYPE=chat_completions to use local vLLM model
JUDGE_API_TYPE = os.getenv("FINQA_JUDGE_API_TYPE", "responses")  # "responses" or "chat_completions"
JUDGE_API_BASE_URL = os.getenv("FINQA_JUDGE_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
JUDGE_API_KEY = os.getenv("FINQA_JUDGE_API_KEY", os.getenv("ARK_API_KEY", ""))
JUDGE_MODEL = os.getenv("FINQA_JUDGE_MODEL", "doubao-seed-2-0-mini-260215")

custom_http_client = httpx.Client(
    http2=True,
    limits=httpx.Limits(max_connections=5000, max_keepalive_connections=3000),
    timeout=100.0,
    trust_env=True,
)

try:
    if not JUDGE_API_KEY:
        raise ValueError("Missing FINQA_JUDGE_API_KEY or ARK_API_KEY")

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

# Step penalty: final_reward = judge_reward - num_steps * STEP_PENALTY_COEF
# Set to 0.0 (or comment out the two "apply step penalty" lines below) to disable.
STEP_PENALTY_COEF = 0.01


def _compute_tool_success_rates(action: str) -> dict[str, dict[str, int]]:
    """Determine successes vs failures per tool by inspecting the ReAct trajectory."""
    stats = {}
    parts = action.split('<tool_call>')
    
    for part in parts[1:]:
        end_idx = part.find('</tool_call>')
        if end_idx == -1:
            continue
            
        call_json_str = part[:end_idx].strip()
        is_parse_error = False
        try:
            call_dict = json.loads(call_json_str)
            tool_name = call_dict.get("name", "unknown")
            if not isinstance(tool_name, str) or not tool_name.strip():
                tool_name = "unknown"
        except Exception:
            tool_name = "format_error"
            is_parse_error = True
            
        if tool_name not in stats:
            stats[tool_name] = {"success": 0, "fail": 0}
            
        if is_parse_error:
            stats[tool_name]["fail"] += 1
            continue
            
        remainder = part[end_idx + len('</tool_call>'):]
        
        # Determine failure by looking for standard fin_qa_tools error signatures
        # "Error:", "Error :", "Error evaluating expression"
        resp_start = remainder.find('<tool_response>')
        resp_end = remainder.find('</tool_response>')
        
        if resp_start != -1 and resp_end != -1:
            response_text = remainder[resp_start:resp_end]
        else:
            response_text = remainder # Fallback to checking the entire chunk between tool calls
            
        if "Error:" in response_text or "Error :" in response_text or "Error evaluating expression" in response_text:
            stats[tool_name]["fail"] += 1
        else:
            stats[tool_name]["success"] += 1
            
    return stats


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
        print("[finqa_reward] JUDGE_MODEL is not set; fallback reward path used.")
        return (False if not is_multi_table else 0.0), {}, judge_stats

    if is_multi_table:
        user_prompt = (
            f"{user_prompt}\n\n"
            "Important: Return valid JSON only and include all required scoring fields."
        )

    try:
        last_exception = None
        for attempt in range(3):
            try:
                if JUDGE_API_TYPE == "chat_completions":
                    # Local vLLM / OpenAI-compatible Chat Completions API
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    response = JUDGE_CLIENT.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    judge_output = response.choices[0].message.content or ""
                    # Strip <think>...</think> blocks from thinking models
                    if "</think>" in judge_output:
                        judge_output = judge_output.split("</think>", 1)[-1].strip()
                else:
                    # Doubao Responses API (default)
                    input_messages = [
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                    ]
                    reasoning_effort = "low" if is_multi_table else "minimal"
                    resp = JUDGE_CLIENT.responses.create(
                        model=JUDGE_MODEL,
                        input=input_messages,
                        reasoning={"effort": reasoning_effort},
                    )
                    judge_output = ""
                    if resp and resp.output:
                        for block in resp.output:
                            if hasattr(block, "content") and block.content:
                                for item in block.content:
                                    if hasattr(item, "text") and item.text:
                                        judge_output += item.text
                break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < 2:
                    import random
                    sleep_time = (attempt * 2) + random.uniform(1.0, 5.0)
                    time.sleep(sleep_time)
                    print(f"[finqa_reward] Retry {attempt + 1} after {sleep_time:.2f}s due to {type(e).__name__}")
                    continue
                raise

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

    is_negative_sample = (ground_truth == "[DATA_UNAVAILABLE]")
    is_negative_single = curriculum_stage == "negative_single_table"
    is_negative_multi = curriculum_stage == "negative_multi_table"

    # ----- Negative sample reward (string matching, no LLM judge) -----
    if is_negative_sample:
        final_answer = _extract_final_answer(action)
        is_correct = "DATA_UNAVAILABLE" in final_answer.upper()
        correctness_reward = 1.0 if is_correct else 0.0

        # Step counting (for logging only, no penalty applied)
        env_num_steps = task_info.get("num_tool_calls", 0)
        text_num_steps = action.count("<tool_call>")
        num_steps = env_num_steps + text_num_steps

        metadata = {
            "correctness_reward": correctness_reward,
            "right_table_access_reward": 0.0,
            "is_single_table_sample": 0.0,
            "is_multi_table_sample": 0.0,
            "is_multi_table_medium_sample": 0.0,
            "is_multi_table_hard_sample": 0.0,
            "is_negative_single_table_sample": 1.0 if is_negative_single else 0.0,
            "is_negative_multi_table_sample": 1.0 if is_negative_multi else 0.0,
            "num_tool_calls": float(num_steps),
            "judge_request_ok": 0.0,
            "judge_client_unavailable": 0.0,
            "judge_timeout_error": 0.0,
            "judge_api_error": 0.0,
            "judge_parse_error": 0.0,
            "judge_fallback_error": 0.0,
        }

        # Initialize ALL tool metric keys to 0.0 (same pattern as positive samples)
        ALL_TOOLS = ["sql_query", "get_table_info", "get_table_names", "calculator", "format_error"]
        for prefix_name in ["single_table", "multi_table", "negative_single_table", "negative_multi_table"]:
            metadata[f"{prefix_name}_num_steps"] = 0.0
            for t_name in ALL_TOOLS:
                metadata[f"{prefix_name}_{t_name}_success_rate"] = 0.0
                metadata[f"{prefix_name}_{t_name}_calls"] = 0.0

        # Populate actual values for this negative sample
        prefix = "negative_single_table" if is_negative_single else "negative_multi_table"
        metadata[f"{prefix}_num_steps"] = float(num_steps)

        env_tool_stats = task_info.get("tool_stats", {})
        text_tool_stats = _compute_tool_success_rates(action)
        merged_stats = {t_name: {"success": 0, "fail": 0} for t_name in ALL_TOOLS}
        for _stats in (env_tool_stats, text_tool_stats):
            for tool_name, counts in _stats.items():
                if tool_name in merged_stats:
                    merged_stats[tool_name]["success"] += counts.get("success", 0)
                    merged_stats[tool_name]["fail"] += counts.get("fail", 0)
        for tool_name, counts in merged_stats.items():
            total = counts['success'] + counts['fail']
            if total > 0:
                success_rate = counts['success'] / float(total)
                metadata[f"{prefix}_{tool_name}_success_rate"] = float(success_rate)
                metadata[f"{prefix}_{tool_name}_calls"] = float(total)

        return RewardOutput(
            reward=correctness_reward - num_steps * STEP_PENALTY_COEF,  # apply step penalty
            is_correct=is_correct,
            metadata=metadata,
        )

    # ----- Positive sample reward (LLM judge) -----
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

    # Step counting (for logging only, no penalty applied)
    env_num_steps = task_info.get("num_tool_calls", 0)
    text_num_steps = action.count("<tool_call>")
    num_steps = env_num_steps + text_num_steps

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
        "is_negative_single_table_sample": 0.0,
        "is_negative_multi_table_sample": 0.0,
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
    }

    # Initialize ALL metric keys statically with 0.0 to prevent Verl logging framework
    # from dropping dynamic keys (if the first batch element didn't have them)
    # and to ensure properly sized mean calculations.
    ALL_TOOLS = ["sql_query", "get_table_info", "get_table_names", "calculator", "format_error"]
    
    for prefix_name in ["single_table", "multi_table", "negative_single_table", "negative_multi_table"]:
        metadata[f"{prefix_name}_num_steps"] = 0.0
        for t_name in ALL_TOOLS:
            metadata[f"{prefix_name}_{t_name}_success_rate"] = 0.0
            metadata[f"{prefix_name}_{t_name}_calls"] = 0.0

    # Populate the actual values for the current instance
    prefix = "multi_table" if is_multi_table else "single_table"
    metadata[f"{prefix}_num_steps"] = float(num_steps)
    
    if is_multi_table:
        # Add all rubric scores to metadata
        for key in CORRECTNESS_WEIGHTS.keys():
            score = rubric.get(key)
            if isinstance(score, int | float):
                metadata[f"multi_table_{key}"] = float(score)
        metadata["multi_table_overall_score"] = correctness_reward

    env_tool_stats = task_info.get("tool_stats", {})
    text_tool_stats = _compute_tool_success_rates(action)
    
    # Merge historical env metrics with final turn text metrics (e.g. format_error)
    merged_stats = {t_name: {"success": 0, "fail": 0} for t_name in ALL_TOOLS}
    
    for _stats in (env_tool_stats, text_tool_stats):
        for tool_name, counts in _stats.items():
            if tool_name in merged_stats:
                merged_stats[tool_name]["success"] += counts.get("success", 0)
                merged_stats[tool_name]["fail"] += counts.get("fail", 0)

    for tool_name, counts in merged_stats.items():
        total = counts['success'] + counts['fail']
        if total > 0:
            success_rate = counts['success'] / float(total)
            metadata[f"{prefix}_{tool_name}_success_rate"] = float(success_rate)
            metadata[f"{prefix}_{tool_name}_calls"] = float(total)

    return RewardOutput(
        reward=correctness_reward - num_steps * STEP_PENALTY_COEF,  # apply step penalty
        is_correct=is_correct,
        metadata=metadata,
    )
