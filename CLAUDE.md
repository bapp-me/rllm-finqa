# FinQA Project

Financial question-answering system that trains a ReAct-style tool-using agent via reinforcement learning (GRPO) to answer quantitative questions about SEC 10-K filings. UC Berkeley rLLM team + Snorkel AI collaboration.

**Core result:** rLLM-FinQA-4B achieves 59.7% Pass@1 on Snorkel FinQA benchmark, beating Qwen3-235B (51.4%) and matching Gemini 2.5 Pro (60.6%). Total training cost <$500.

## Architecture

```
train_finqa.py / train_finqa_tinker.py   (training entry points)
  ├── FinQAAgent (fin_qa_agent.py)       — ReAct agent with 4 tools
  │     └── prompts/react_system_prompt.txt
  ├── FinQAEnvironment (fin_qa_environment.py) — tool execution + reward
  │     └── fin_qa_reward_doubao.py      (active reward, Doubao judge)
  ├── fin_qa_tools.py                    — GetTableNames, GetTableInfo, SQLQuery, Calculator
  │     └── data/company_tables/         (in-memory SQLite at import)
  ├── finqa_curriculum_sampler.py         — adaptive sampling (single→multi-table)
  └── prepare_finqa_data.py              — HF dataset download + curriculum splits
```

**Reward variants:** `fin_qa_reward.py` (Qwen/Dashscope), `fin_qa_reward_qwen.py` (+ efficiency penalty), `fin_qa_reward_doubao.py` (Doubao/Volcengine, currently active).

## Key Commands

```bash
# Data preparation
python -m projects.finqa.prepare_finqa_data

# Training (verl backend, 8 GPUs, Qwen3-4B)
export FINQA_REWARD_API_KEY=... FINQA_REWARD_API_BASE_URL=... FINQA_REWARD_MODEL=...
bash projects/finqa/train_finqa.sh

# Training (Tinker backend, Qwen3-30B LoRA)
bash projects/finqa/train_finqa_tinker.sh

# Inference
python -m vllm.entrypoints.openai.api_server --model rLLM/rLLM-FinQA-4B --port 30000 --dtype bfloat16
python -m projects.finqa.run_finqa
```

## Data

- **Source:** `rLLM/finqa` on HuggingFace — 5,110+ Q&A pairs, 207 companies, 6,923 tables
- **Splits:** 165 train / 20 val / 22 test companies (no overlap)
- **Types:** single-table, multi-table (medium/hard), negative samples (unanswerable → `[DATA_UNAVAILABLE]`)
- **Paths:** defined in `constants.py`, `data/` directory

## Data Generation Pipeline (offline)

```
scripts/data_generation/download_10k.py  → SEC EDGAR scraping (Selenium)
scripts/data_generation/cleanup_tables.py → table structuring (Qwen3-30B)
scripts/data_generation/generate_questions.py → Q&A generation + verification
scripts/generate_negative_samples.py → unanswerable question generation (Qwen3-235B)
```

## Dependencies

Core: `asteval`, `openai`, `httpx[http2]`. Parent `rllm` framework provides `ToolAgent`, `ToolEnvironment`, `AgentTrainer`, `MultiTurnWorkflow`, `DatasetRegistry`. Also requires `pandas`, `numpy`, `torch`, `transformers`, `hydra-core`, `omegaconf`, `verl`.

## Key Config

- **Models:** Qwen3-4B-Instruct-2507 (verl) or Qwen3-30B-A3B-Instruct-2507 (Tinker/LoRA)
- **Reward:** Single-table = binary judge; Multi-table = rubric-based (4 dimensions, threshold ≥0.9); Negative = string match
- **Training (shell script):** batch 384, max prompt 3072, max response 16384, GRPO group_size 8, lr 1e-6, temp 0.7, 5 epochs
- **Training (blog report):** batch 256, max prompt 2048, 10 epochs, 120 steps total, ~21h on 8xH100
- **Judge:** blog uses gpt-5-nano; code currently uses Doubao (doubao-seed-2-0-mini-260215). Blog mentions Portkey caching (~40% hit rate, ~$40 judge cost)
- **Env vars:** `FINQA_REWARD_API_KEY`, `FINQA_REWARD_API_BASE_URL`, `FINQA_REWARD_MODEL`, optional `FINQA_TABLES_ROOT`

## Key Findings (from official blog)

**"Terence Tao" paradox:** Task success depends on disciplined, repeatable tool use, not open-ended reasoning. Large models fail due to:
- Schema hallucination — guessing table/column names instead of using `get_table_names()`/`get_table_info()`
- Context flooding — running `SELECT *` queries that overflow context
- No error recovery — repeating failed strategies instead of reading error messages

**Ablation 1: Simple data teaches complex skills**
- Single-table only training (66.3%) > single+multi-table (61.6%) > curriculum (64.8%)
- Tool-use discipline learned on simple tasks generalizes to multi-table reasoning without explicit training

**Ablation 2: Binary reward beats partial reward**
- Binary correctness (66.3%) > partial reward with intermediate step credit (54.0%)
- Base model already finds correct tables ~73% of the time; bottleneck is SQL/calculation errors (~62%)
- Partial rewards let the model "game" easy sub-tasks; sparse binary signal forces optimizing the full trajectory

**Training dynamics:** Rapid adaptation in first 60 steps (50% → 72% val accuracy), then plateau.

**No generalization loss:** BFCL benchmark shows financial specialization doesn't hurt general tool-calling (35.65% vs 35.02% base).

## Benchmark Results

| Model | FinQA (290) | FinQA-Reasoning (79) |
|-------|-------------|----------------------|
| Qwen3-4B base | 27.9% | 13.9% |
| gpt-5-nano | 50.0% | 26.6% |
| Qwen3-235B | 51.4% | 18.9% |
| **rLLM-FinQA-4B** | **59.7%** | **26.6%** |
| Gemini 2.5 Pro | 60.6% | 34.6% |
| GPT-4.1 | 62.7% | 37.9% |
