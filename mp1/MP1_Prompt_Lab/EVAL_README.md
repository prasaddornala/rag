Evaluation pipeline for MP1 Prompt Lab

Usage:

1. Install dependencies from `requirements.txt` (prefer a venv):

```bash
pip install -r requirements.txt
```

2. Run in mock mode (no OpenAI key required):

```bash
python eval_pipeline.py --mode mock --out results
```

3. Run using OpenAI (set `OPENAI_API_KEY`) to call the API and collect latency/token usage:

```bash
export OPENAI_API_KEY="sk-..."
python eval_pipeline.py --mode openai --model gpt-3.5-turbo --out results_openai

Judge mode and cost accounting:
- Use `--judge` to enable an LLM-as-a-Judge scoring pass. When enabled the pipeline will ask the LLM to score each candidate output vs the golden answer per field (0/0.5/1). In mock mode a deterministic local judge is used.\n
- The pipeline aggregates token usage for both extraction and judge calls and computes a simple USD cost estimate using a per-model price map embedded in the script. The results JSON includes `metrics.total_tokens` and `metrics.total_cost_usd` per strategy.
```

What it does:
- Implements four prompting strategies: `simple`, `schema`, `fewshot`, `retrieval`.
- Loads `data/job_snippets.jsonl` and `data/golden_set.jsonl`.
- Runs extraction for each snippet, computes per-field accuracy and an overall accuracy metric, and saves `results.json`.

Notes:
- The script supports a mock deterministic extractor when OpenAI key is not set; this helps smoke-test the pipeline offline.
- The LLM-as-a-Judge extension and more advanced cost accounting can be added if you have an API key and would like me to implement it.
