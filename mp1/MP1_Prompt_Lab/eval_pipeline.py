#!/usr/bin/env python3
import os
import time
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import openai
except Exception:
    openai = None

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # try to find the first JSON object in text
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # fallback: simple key:value parsing
    obj = {}
    for k in ["company", "role", "years"]:
        pat = re.compile(rf"{k}[:\s]*([\w\- &.,]+)", flags=re.I)
        mm = pat.search(text)
        if mm:
            obj[k] = mm.group(1).strip()
    return obj if obj else None

def simple_rule_extract(snippet: str) -> Dict[str, Any]:
    company = None
    role = None
    years = None
    # company: look for 'Company: X' or capitalized words followed by 'is hiring' or 'at'
    m = re.search(r"Company[:\s]+([A-Z][\w &.,-]+)", snippet)
    if m:
        company = m.group(1).strip()
    else:
        m = re.search(r"at\s+([A-Z][\w &.,-]+)", snippet)
        if m:
            company = m.group(1).strip()
    m = re.search(r"(Senior|Junior|Lead|Manager|Engineer|Developer|Scientist|Analyst)[\w\s,-]*", snippet)
    if m:
        role = m.group(0).strip()
    m = re.search(r"(\d+)\+?\s*(?:years|yrs)", snippet, flags=re.I)
    if m:
        years = m.group(1)
    return {"company": company, "role": role, "years": years}

def call_openai_chat(system: str, prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
    if openai is None:
        raise RuntimeError("openai package not installed")
    start = time.perf_counter()
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    latency = time.perf_counter() - start
    content = resp.choices[0].message.content
    usage = resp.get("usage", {})
    return content, {"latency": latency, "usage": usage}

def build_prompt(strategy: str, snippet: str, examples: Optional[List[Dict[str, Any]]] = None) -> str:
    if strategy == "simple":
        return (
            "Extract the following fields from the job posting snippet: Company (name), Role, Minimum years of experience required. "
            "Return a JSON object with keys \"company\", \"role\", \"years\".\n\nSnippet:\n" + snippet
        )
    if strategy == "schema":
        return (
            "Extract the fields and return strictly JSON matching schema: {\"company\": string|null, \"role\": string|null, \"years\": int|null}. "
            "If a value is not present, use null. Provide only the JSON object.\n\nSnippet:\n" + snippet
        )
    if strategy == "fewshot":
        ex_text = "\n\n".join([json.dumps(e, ensure_ascii=False) + "\nExample snippet: " + e.get("snippet", "") for e in (examples or [])])
        return (
            "You are given snippets and desired JSON outputs. Follow the examples exactly. Return only JSON.\n\n" + ex_text + "\n\nNow process:\n" + snippet
        )
    if strategy == "retrieval":
        return (
            "You are an information extraction assistant. Use the snippet below as the source of truth. Extract company name, job role, and minimum years of experience. "
            "If the snippet is ambiguous, answer conservatively and put null for unknowns. Output a strict JSON object with keys company, role, years.\n\nSnippet:\n" + snippet
        )
    raise ValueError("unknown strategy")

def evaluate_predictions(preds: List[Dict[str, Any]], gold: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(preds) == len(gold)
    totals = {"company": 0, "role": 0, "years": 0}
    parsed = 0
    for p, g in zip(preds, gold):
        if p is None:
            continue
        parsed += 1
        for k in totals.keys():
            pv = p.get(k)
            gv = g.get(k) or g.get(k.lower())
            if pv is None:
                continue
            if isinstance(pv, str) and isinstance(gv, str) and pv.strip().lower() == gv.strip().lower():
                totals[k] += 1
            else:
                # try numeric compare for years
                try:
                    if k == "years" and gv is not None and pv is not None:
                        if int(pv) == int(gv):
                            totals[k] += 1
                except Exception:
                    pass
    acc = {k: (totals[k] / len(preds)) for k in totals}
    overall = sum(totals.values()) / (len(preds) * len(totals))
    return {"field_counts": totals, "accuracies": acc, "overall_accuracy": overall, "parsed_count": parsed}


def simple_judge(pred: Optional[Dict[str, Any]], gold: Dict[str, Any]) -> Dict[str, float]:
    # deterministic local scoring: 1 for exact match, 0.5 for near match (years off by 1), 0 otherwise
    scores = {"company": 0.0, "role": 0.0, "years": 0.0}
    if pred is None:
        return scores
    for k in scores.keys():
        pv = pred.get(k)
        gv = gold.get(k) or gold.get(k.lower())
        if pv is None or gv is None:
            continue
        try:
            if k == "years":
                if int(pv) == int(gv):
                    scores[k] = 1.0
                elif abs(int(pv) - int(gv)) == 1:
                    scores[k] = 0.5
            else:
                if isinstance(pv, str) and isinstance(gv, str) and pv.strip().lower() == gv.strip().lower():
                    scores[k] = 1.0
        except Exception:
            # fallback string compare
            if isinstance(pv, str) and isinstance(gv, str) and pv.strip().lower() == gv.strip().lower():
                scores[k] = 1.0
    return scores


def llm_judge_call(pred: Dict[str, Any], gold: Dict[str, Any], model: str = "gpt-3.5-turbo") -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """Call the LLM to judge a single prediction against gold. Returns (scores_dict, usage_info)."""
    if openai is None:
        return None, {}
    system = "You are an objective evaluator that scores extracted fields from job postings."
    prompt = (
        "Gold JSON:\n" + json.dumps(gold, ensure_ascii=False) + "\n\n"
        "Candidate JSON:\n" + json.dumps(pred, ensure_ascii=False) + "\n\n"
        "Rubric: For each field (company, role, years) return a numeric score between 0 and 1. "
        "Use 1 for exact match, 0.5 for partial/near match, 0 for incorrect or missing. "
        "Return ONLY a JSON object with keys company, role, years and optional notes. Example: {\"company\":1,\"role\":0.5,\"years\":1,\"notes\":\"reason\"}."
    )
    try:
        text, info = call_openai_chat(system, prompt, model=model, temperature=0.0)
    except Exception as e:
        return None, {}
    parsed = parse_json_from_text(text)
    # normalize parsed scores to floats if present
    if isinstance(parsed, dict):
        scores = {}
        for k in ["company", "role", "years"]:
            v = parsed.get(k)
            try:
                scores[k] = float(v) if v is not None else 0.0
            except Exception:
                scores[k] = 0.0
        return scores, info
    return None, info


def compute_cost(total_tokens: int, model: str) -> float:
    # simple cost map per 1k tokens (USD)
    price_map = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03,
        "gpt-4o": 0.03,
    }
    price_per_1k = price_map.get(model, 0.002)
    return (total_tokens / 1000.0) * price_per_1k

def run_pipeline(mode: str = "mock", model: str = "gpt-3.5-turbo", judge: bool = False, out_dir: Optional[str] = None):
    base = Path(__file__).parent
    data_snippets = load_jsonl(base / "data" / "job_snippets.jsonl")
    golden = load_jsonl(base / "data" / "golden_set.jsonl")
    strategies = ["simple", "schema", "fewshot", "retrieval"]
    results = {}
    for strat in strategies:
        preds = []
        metrics = {"calls": 0, "total_latency": 0.0, "total_tokens": 0, "judge_calls": 0, "judge_tokens": 0}
        judge_agg = {"company": 0.0, "role": 0.0, "years": 0.0}
        for i, item in enumerate(data_snippets):
            snippet = item.get("snippet") or item.get("text") or item.get("job_snippet") or item.get("content") or item.get("sentence")
            if strat == "fewshot":
                # take up to 2 examples from golden
                examples = []
                for ex in golden[:2]:
                    ex_copy = {"company": ex.get("company"), "role": ex.get("role"), "years": ex.get("years"), "snippet": ex.get("snippet", ex.get("text", ""))}
                    examples.append(ex_copy)
                prompt = build_prompt(strat, snippet, examples=examples)
            else:
                prompt = build_prompt(strat, snippet)

            if mode == "mock" or openai is None or os.environ.get("OPENAI_API_KEY") is None:
                # deterministic local extractor
                out = simple_rule_extract(snippet)
                preds.append(out)
                # judge locally if requested
                if judge:
                    sc = simple_judge(out, golden[i])
                    for k in judge_agg:
                        judge_agg[k] += sc.get(k, 0.0)
            else:
                system = "You are a concise JSON extractor."
                try:
                    text, info = call_openai_chat(system, prompt, model=model)
                except Exception as e:
                    print("LLM call failed:", e)
                    preds.append(None)
                    if judge:
                        # count failed judge as zero
                        for k in judge_agg:
                            judge_agg[k] += 0.0
                    continue
                metrics["calls"] += 1
                metrics["total_latency"] += info.get("latency", 0.0)
                usage = info.get("usage", {})
                metrics["total_tokens"] += usage.get("total_tokens", 0)
                parsed = parse_json_from_text(text)
                if parsed is None:
                    parsed = parse_json_from_text(text.replace("\n", " "))
                preds.append(parsed)
                # optionally run LLM-as-a-Judge on the candidate
                if judge:
                    scores, jinfo = llm_judge_call(parsed, golden[i], model=model)
                    metrics["judge_calls"] += 1
                    if jinfo:
                        jusage = jinfo.get("usage", {})
                        metrics["judge_tokens"] += jusage.get("total_tokens", 0)
                    if scores is None:
                        # if judge failed, fall back to simple_judge
                        scores = simple_judge(parsed, golden[i])
                    for k in judge_agg:
                        judge_agg[k] += scores.get(k, 0.0)
        evalr = evaluate_predictions(preds, golden)
        # compute cost (includes extraction + judge tokens)
        total_tokens = metrics.get("total_tokens", 0) + metrics.get("judge_tokens", 0)
        total_cost = compute_cost(total_tokens, model)
        metrics["total_tokens"] = total_tokens
        metrics["total_cost_usd"] = total_cost

        judge_summary = None
        if judge:
            n = len(data_snippets) if len(data_snippets) > 0 else 1
            judge_summary = {k: (judge_agg[k] / n) for k in judge_agg}
            judge_summary["overall"] = sum(judge_summary[k] for k in ["company", "role", "years"]) / 3.0

        results[strat] = {"preds": preds, "metrics": metrics, "eval": evalr, "judge_summary": judge_summary}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mock", "openai"], default="mock")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--judge", action="store_true", help="Enable LLM-as-a-Judge scoring (or mock judge in mock mode)")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    res = run_pipeline(mode=("openai" if args.mode == "openai" else "mock"), model=args.model, judge=args.judge, out_dir=args.out)
    print(json.dumps({k: {"overall": v["eval"]["overall_accuracy"], "accuracies": v["eval"]["accuracies"]} for k, v in res.items()}, indent=2))

if __name__ == "__main__":
    main()
