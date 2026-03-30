import os
import time
import json
import logging
import requests
import ast
import pandas as pd
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Ensure projects folder is in path so we can import fin_qa_tools properly
# fin_qa_tools has relative imports like 'from .constants import', so we MUST import it as a package module.
# To do this, we need the parent of 'rllm' in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # This is the finqa directory
# The project root is three levels up from finqa (finqa -> projects -> rllm -> parent_of_rllm)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from rllm.projects.finqa.fin_qa_tools import GetTableInfo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Existing API Client extracted from user's files ---
CONFIG = {
    "model_name": "qwen-235b",
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model_path": "/merchant_reco_l3/wangmuxuan/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
    "max_tokens": 12288,
    "temperature": 0.1,
    "top_p": 0.9,
    "timeout": 800,
    "retry_times": 3,
    "retry_delay": 10.0,
}

class VLLMClient:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config.get("api_url")
        self._model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.max_tokens = config.get("max_tokens", 8192)
        self.temperature = config.get("temperature", 0.1)
        self.top_p = config.get("top_p", 0.9)
        self.timeout = config.get("timeout", 300)
        self.retry_times = config.get("retry_times", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
    
    def chat(self, system_prompt: str, user_prompt: str) -> Tuple[bool, str, str]:
        """Returns success_bool, think_content, answer_content"""
        payload = {
            "model": self.model_path or self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.retry_times):
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                content = data['choices'][0]['message']['content']
                think, ans = self._extract_think_answer(content)
                return True, think, ans
            except Exception as e:
                logging.warning(f"Request failed (attempt {attempt+1}/{self.retry_times}): {e}")
                time.sleep(self.retry_delay)
        return False, "", ""
    
    def _extract_think_answer(self, content: str) -> tuple:
        split_tag = "</think>"
        if split_tag in content:
            parts = content.split(split_tag)
            think_content = parts[0].replace("<think>", "").strip()
            answer_content = parts[1].strip()
            return think_content, answer_content
        return "", content

# --- Prompts ---
BASE_DIR = "/Users/henrykqin/Documents/代码/rllm/projects/finqa"
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

with open(os.path.join(PROMPTS_DIR, "negative_question_generation_prompt.txt"), "r", encoding="utf-8") as f:
    GENERATE_NEGATIVE_PROMPT = f.read()

with open(os.path.join(PROMPTS_DIR, "negative_question_verification_prompt.txt"), "r", encoding="utf-8") as f:
    VERIFY_NEGATIVE_PROMPT = f.read()

# --- Main Logic ---
def process_single_row(idx: int, row: pd.Series, client: VLLMClient, sample_size: int, is_multi: bool) -> dict | None:
    """Process a single row to generate and verify a negative sample."""
    original_q = row.get("question", "")
    original_uq = row.get("user_query", "")
    if "core_question" in row and pd.notna(row["core_question"]):
        original_q = row["core_question"]
    
    context_info = f"Original Question: {original_q}\n"
    context_info += f"Original User Query: {original_uq}\n"
    context_info += f"Original Answer: {row.get('ground_truth', 'N/A')}\n"
    
    # Extract table names and fetch schema info
    company_name = row.get("company", "")
    table_name_raw = row.get("table_name", "")
    
    table_names = []
    if isinstance(table_name_raw, str):
        try:
            # Multi-table might be saved as a string representation of a list
            table_names = ast.literal_eval(table_name_raw)
            if not isinstance(table_names, list):
                table_names = [table_name_raw]
        except (ValueError, SyntaxError):
            table_names = [table_name_raw]
    elif isinstance(table_name_raw, list):
        table_names = table_name_raw

    table_schemas = []
    table_info_tool = GetTableInfo()
    for tn in table_names:
        schema = table_info_tool.get_table_info(company_name, tn)
        if not schema.startswith("Error"):
            table_schemas.append(f"Table Name: {tn}\nSchema and Samples:\n{schema}")
    
    schema_context = "\n\n".join(table_schemas)
    if not schema_context:
        schema_context = "No detailed schema available."
        
    context_info += f"\n--- PROVIDED TABLE SCHEMA(S) ---\n{schema_context}\n--------------------------------\n"
    
    # 1. Generate Negative Question
    logging.info(f"[{idx+1}/{sample_size}] Generating negative for: {original_q}")
    success, think, ans = client.chat(GENERATE_NEGATIVE_PROMPT, context_info)
    
    if not success:
        logging.error(f"Failed to generate for row {idx}")
        return None

    try:
        # Clean possible markdown formatting
        ans_clean = ans.replace("```json", "").replace("```", "").strip()
        num_start = ans_clean.find("{")
        num_end = ans_clean.rfind("}") + 1
        if num_start != -1 and num_end != 0:
             ans_clean = ans_clean[num_start:num_end]
             
        parsed_gen = json.loads(ans_clean)
        neg_q = parsed_gen.get("question", "")
        if not neg_q:
             logging.warning(f"Generated JSON missing 'question' key for row {idx}")
             return None
    except Exception as e:
        logging.error(f"Failed to parse generation JSON: {e}\nOutput was: {ans}")
        return None

    # 2. Verify Negative Question
    verify_context = f"Question to verify: {neg_q}\n"
    verify_context += f"\n--- PROVIDED TABLE SCHEMA(S) ---\n{schema_context}\n--------------------------------\n"
    verify_context += f"\n(Hint: The original valid question was '{original_q}')\n"
    verify_context += "Determine if this new question requires data NOT present based on the original question's scope."

    logging.info(f"[{idx+1}/{sample_size}] Verifying generated question: {neg_q}")
    v_success, v_think, v_ans = client.chat(VERIFY_NEGATIVE_PROMPT, verify_context)
    
    if not v_success:
         logging.error(f"Verification API call failed for row {idx}.")
         return None
         
    try:
        v_clean = v_ans.replace("```json", "").replace("```", "").strip()
        v_start = v_clean.find("{")
        v_end = v_clean.rfind("}") + 1
        if v_start != -1 and v_end != 0:
             v_clean = v_clean[v_start:v_end]
             
        parsed_ver = json.loads(v_clean)
        can_answer = parsed_ver.get("can_answer", True)
        reason = parsed_ver.get("reason", "")
    except Exception as e:
        logging.error(f"Failed to parse verification JSON for row {idx}: {e}")
        return None

    if can_answer == False:
        logging.info(f"[{idx+1}/{sample_size}] Valid negative sample created! Reason: {reason}")
        
        # Create new row
        new_row = row.copy()
        
        # If model generated a new user_query that fits the format, use it. Otherwise fallback.
        gen_uq = parsed_gen.get("user_query", "")
        if gen_uq:
            new_row["user_query"] = gen_uq
        else:
            if "core_question" in new_row and pd.notna(new_row["core_question"]):
                company = new_row.get("company", "")
                if is_multi:
                     new_row["user_query"] = f"For company `{company}`, here is the question:\n\nQuestion:\n{neg_q}"
                else:
                     new_row["user_query"] = f"For company `{company}`, here is the question: {neg_q}"
            else:
                 new_row["user_query"] = neg_q

        if "core_question" in new_row and pd.notna(new_row["core_question"]):
            new_row["core_question"] = neg_q
            if is_multi:
                 new_row["question"] = f"Use the tools to answer the following multi-table question: {neg_q}"
            else:
                 new_row["question"] = f"Answer the following question based on the provided table: {neg_q}"
        else:
            new_row["question"] = neg_q
        
        new_row["ground_truth"] = "[DATA_UNAVAILABLE]"
        
        # Override original answer and explanation so they don't leak from the positive sample
        if "answer" in new_row:
             new_row["answer"] = "[DATA_UNAVAILABLE]"
        if "explanation" in new_row:
             new_row["explanation"] = parsed_gen.get("explanation", reason)
             
        q_type = new_row.get("question_type", "")
        if is_multi:
            new_row["question_type"] = f"{q_type}_negative" if q_type else "multi_table_negative"
        else:
            new_row["question_type"] = f"{q_type}_negative" if q_type else "single_table_negative"
            
        return new_row.to_dict()
    else:
         logging.warning(f"[{idx+1}/{sample_size}] Verification rejected negative sample (answerable). Reason: {reason}")
         return None

def create_negative_samples(input_csv: str, output_csv: str, sample_size: int = 100, is_multi: bool = False, max_workers: int = 160):
    logging.info(f"Loading data from {input_csv} (is_multi={is_multi})")
    df = pd.read_csv(input_csv)
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).copy()
    else:
        df = df.copy()

    client = VLLMClient(CONFIG)
    new_rows = []
    
    # We must instantiate but we don't need to call _preload_all() here explicitly
    # because fin_qa_tools automatically runs it on import.
    
    logging.info(f"Starting concurrent generation with max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, idx, row, client, len(df), is_multi): idx
            for idx, row in df.iterrows()
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                new_rows.append(result)

    # Output Data
    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(output_csv, index=False)
    logging.info(f"Successfully saved {len(out_df)} negative samples to {output_csv}")

if __name__ == "__main__":
    data_dir = os.path.join(BASE_DIR, "example/data")
    
    # Generate Single Table Negatives
    single_table_in = os.path.join(data_dir, "val_finqa.csv")
    single_table_out = os.path.join(data_dir, "val_finqa_negative.csv")
    
    # Generate Multi Table Negatives
    multi_table_in = os.path.join(data_dir, "multi_table_data/val_finqa.csv")
    multi_table_out = os.path.join(data_dir, "multi_table_data/val_finqa_negative.csv")
    
    if os.path.exists(single_table_in):
        create_negative_samples(single_table_in, single_table_out, sample_size=100, is_multi=False)
    else:
        logging.error(f"Cannot find {single_table_in}")
        
    if os.path.exists(multi_table_in):
        create_negative_samples(multi_table_in, multi_table_out, sample_size=100, is_multi=True)
    else:
        logging.error(f"Cannot find {multi_table_in}")
