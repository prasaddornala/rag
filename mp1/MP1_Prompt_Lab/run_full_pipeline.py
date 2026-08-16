import asyncio
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI
import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


MODEL = os.getenv('MODEL', 'gpt-4o-mini')
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'gpt-4o')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
assert OPENAI_API_KEY, 'OPENAI_API_KEY must be set in .env or environment'

RATES = {
    'gpt-4o-mini': {'in': 0.15 / 1_000_000, 'out': 0.60 / 1_000_000},
    'gpt-4o':      {'in': 2.50 / 1_000_000, 'out': 10.00 / 1_000_000},
}


def load_data():
    snippets = [json.loads(line) for line in (DATA_DIR / 'job_snippets.jsonl').read_text().splitlines() if line.strip()]
    golden = {row['id']: row for row in (json.loads(line) for line in (DATA_DIR / 'golden_set.jsonl').read_text().splitlines() if line.strip())}
    return snippets, golden


def prompt_zero_shot(snippet_text: str) -> list[dict]:
    return [{
        'role': 'user',
        'content': ("Extract these fields from the job posting snippet and return ONLY a JSON object with keys:\n"
                    "company, role, years_experience_required (integer or null).\n\nSnippet:\n" + snippet_text)
    }]


def prompt_few_shot(snippet_text: str) -> list[dict]:
    examples = [
        {'snippet': 'Acme Corp is hiring a Senior Software Engineer — 5+ years experience required.', 'output': {'company': 'Acme Corp', 'role': 'Senior Software Engineer', 'years_experience_required': 5}},
        {'snippet': 'Northwind Ltd: Data Analyst (2 years experience preferred).', 'output': {'company': 'Northwind Ltd', 'role': 'Data Analyst', 'years_experience_required': 2}},
    ]
    example_text = "\n\n".join([f"Snippet: {e['snippet']}\nOutput: {json.dumps(e['output'])}" for e in examples])
    return [{
        'role': 'user',
        'content': ("You are an extractor. Follow the examples exactly and return only a JSON object with keys:"
                    " company, role, years_experience_required.\n\n" + example_text + "\n\nNow process:\n" + snippet_text)
    }]


def prompt_structured(snippet_text: str) -> list[dict]:
    system = {'role': 'system', 'content': ("You are an expert recruiter and a strict JSON formatter.\n"
                                              "Output MUST be a single JSON object with EXACT keys: company, role, years_experience_required.\n"
                                              "- company: string or null\n- role: string or null\n- years_experience_required: integer or null\n")}
    user = {'role': 'user', 'content': "Extract the schema values from the snippet below and return only the JSON object.\n\nSnippet:\n" + snippet_text}
    return [system, user]


def prompt_cot(snippet_text: str) -> list[dict]:
    return [{'role': 'user', 'content': ("Read the snippet and THINK STEP BY STEP about where the company, role, and minimum years are stated.\n"
                                            "After your reasoning, output a final JSON object with keys: company, role, years_experience_required.\n\nSnippet:\n" + snippet_text)}]


STRATEGIES = {
    'zero_shot': prompt_zero_shot,
    'few_shot': prompt_few_shot,
    'structured': prompt_structured,
    'cot': prompt_cot,
}


def parse_response(text: str) -> dict | None:
    if not text or not isinstance(text, str):
        return None
    # remove fences
    text = re.sub(r"```\w*", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    jtxt = m.group(0)
    try:
        obj = json.loads(jtxt)
    except Exception:
        try:
            obj = json.loads(jtxt.replace("'", '"'))
        except Exception:
            return None
    out = {}
    out['company'] = obj.get('company') or obj.get('Company') or obj.get('employer')
    out['role'] = obj.get('role') or obj.get('title') or obj.get('job_title')
    yrs = obj.get('years_experience_required') or obj.get('years') or obj.get('years_experience')
    if isinstance(yrs, str):
        yrs = yrs.strip()
        if yrs.isdigit():
            yrs = int(yrs)
        else:
            m2 = re.search(r"(\d+)", yrs)
            yrs = int(m2.group(1)) if m2 else None
    out['years_experience_required'] = yrs if yrs is not None else None
    return out


def extract_usage(resp) -> tuple[int, int, int]:
    # support dataclass-like and dict-like usage
    usage = getattr(resp, 'usage', None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get('usage', {})
    if usage is None:
        return 0, 0, 0
    # try attributes
    prompt_tokens = getattr(usage, 'prompt_tokens', None) or usage.get('prompt_tokens', None) if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', None)
    completion_tokens = getattr(usage, 'completion_tokens', None) or usage.get('completion_tokens', None) if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)
    total_tokens = getattr(usage, 'total_tokens', None) or usage.get('total_tokens', None) if isinstance(usage, dict) else getattr(usage, 'total_tokens', None)
    prompt_tokens = int(prompt_tokens) if prompt_tokens else 0
    completion_tokens = int(completion_tokens) if completion_tokens else 0
    total_tokens = int(total_tokens) if total_tokens else prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


async def run_one(client, strategy_name: str, snippet: dict) -> dict:
    messages = STRATEGIES[strategy_name](snippet.get('text', snippet.get('snippet', '')))
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(model=MODEL, messages=messages, temperature=0.0)
    latency = time.perf_counter() - t0
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)
    parsed = parse_response(content)
    prompt_tokens, completion_tokens, total_tokens = extract_usage(resp)
    rates = RATES.get(MODEL, {'in': 0.0, 'out': 0.0})
    cost = prompt_tokens * rates['in'] + completion_tokens * rates['out']
    return {
        'strategy': strategy_name,
        'snippet_id': snippet.get('id'),
        'raw': content,
        'parsed': parsed,
        'cost_usd': cost,
        'latency_s': latency,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
    }


async def run_all(client, snippets):
    tasks = []
    for sname in STRATEGIES.keys():
        for snip in snippets:
            tasks.append(run_one(client, sname, snip))
    results = await asyncio.gather(*tasks)
    return results


def score_accuracy(extracted: dict | None, gold: dict) -> int:
    if extracted is None:
        return 0
    score = 0
    def norm(s):
        return (s or '').strip().lower() if isinstance(s, str) else s
    if norm(extracted.get('company')) and norm(gold.get('company')) and norm(extracted.get('company')) == norm(gold.get('company')):
        score += 1
    if norm(extracted.get('role')) and norm(gold.get('role')) and norm(extracted.get('role')) == norm(gold.get('role')):
        score += 1
    ev = extracted.get('years_experience_required')
    gv = gold.get('years_experience_required')
    try:
        if ev is None and gv is None:
            pass
        elif ev is not None and gv is not None and int(ev) == int(gv):
            score += 1
    except Exception:
        pass
    return score


async def score_llm_judge(client, extracted: dict | None, gold: dict) -> int:
    if extracted is None:
        return 1
    system = "You are an objective evaluator. Return a single integer 1-4 following the rubric exactly. Reply with just the integer."
    prompt = (
        "Gold JSON:\n" + json.dumps(gold, ensure_ascii=False) + "\n\n"
        "Candidate JSON:\n" + json.dumps(extracted, ensure_ascii=False) + "\n\n"
        "Rubric:\n4 — all three fields correct\n3 — two of three correct, no fabricated data\n2 — one of three correct, or fabricated a field\n1 — none correct or unparsable\n\nReply with a single integer (1-4)."
    )
    try:
        resp = await client.chat.completions.create(model=JUDGE_MODEL, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}], temperature=0.0)
        txt = resp.choices[0].message.content.strip()
        m = re.search(r"([1-4])", txt)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 1


async def main():
    snippets, golden = load_data()
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    print(f'Running {len(snippets)} snippets × {len(STRATEGIES)} strategies = {len(snippets) * len(STRATEGIES)} calls...')
    results = await run_all(client, snippets)

    # score and run judge calls
    scored = []
    judge_tasks = []
    for r in results:
        gid = r['snippet_id']
        gold_entry = golden.get(gid)
        acc = score_accuracy(r.get('parsed'), gold_entry)
        r['accuracy'] = acc
        r['parse_success'] = 1 if r.get('parsed') is not None else 0
        judge_tasks.append(score_llm_judge(client, r.get('parsed'), gold_entry))
        scored.append(r)

    print('Running judge calls (this will make one judge call per result)')
    judge_scores = await asyncio.gather(*judge_tasks)
    for row, jscore in zip(scored, judge_scores):
        row['llm_judge_score'] = int(jscore)

    out_jsonl = RESULTS_DIR / 'mp1_results.jsonl'
    with out_jsonl.open('w', encoding='utf8') as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    df = pd.DataFrame(scored)
    summary = df.groupby('strategy').agg({'accuracy': 'mean', 'parse_success': 'mean', 'llm_judge_score': 'mean', 'cost_usd': 'sum', 'latency_s': 'median'}).round(3)
    summary.columns = ['Accuracy (mean of 3)', 'Parse rate', 'Judge score', 'Total cost ($)', 'Latency p50 (s)']
    out_md = RESULTS_DIR / 'mp1_comparison.md'
    out_md.write_text('# MP1 comparison\n\n' + summary.to_markdown())

    print('Full run complete. Results written to:', out_jsonl)
    print(summary)


if __name__ == '__main__':
    asyncio.run(main())
