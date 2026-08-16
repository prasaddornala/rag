import asyncio
import json
import random
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    snippets = [json.loads(line) for line in (DATA_DIR / 'job_snippets.jsonl').read_text().splitlines() if line.strip()]
    golden = {row['id']: row for row in (json.loads(line) for line in (DATA_DIR / 'golden_set.jsonl').read_text().splitlines() if line.strip())}
    return snippets, golden


def prompt_zero_shot(snippet_text: str) -> list[dict]:
    return [{'role': 'user', 'content': 'Extract JSON: company, role, years_experience_required.\n\n' + snippet_text}]


def prompt_few_shot(snippet_text: str) -> list[dict]:
    return [{'role': 'user', 'content': 'Examples omitted for dry-run. Extract JSON only.\n\n' + snippet_text}]


def prompt_structured(snippet_text: str) -> list[dict]:
    return [{'role': 'system', 'content': 'Strict JSON output required.'}, {'role': 'user', 'content': snippet_text}]


def prompt_cot(snippet_text: str) -> list[dict]:
    return [{'role': 'user', 'content': 'Think step-by-step then output JSON.\n\n' + snippet_text}]


STRATEGIES = {
    'zero_shot': prompt_zero_shot,
    'few_shot': prompt_few_shot,
    'structured': prompt_structured,
    'cot': prompt_cot,
}


def parse_response(text: str) -> dict | None:
    if not text or not isinstance(text, str):
        return None
    try:
        m = json.loads(text)
        return {
            'company': m.get('company'),
            'role': m.get('role'),
            'years_experience_required': m.get('years_experience_required'),
        }
    except Exception:
        return None


class MockResp:
    def __init__(self, content: str, prompt_tokens: int = 10, completion_tokens: int = 20):
        class Msg:
            def __init__(self, c):
                self.content = c

        class Choice:
            def __init__(self, msg):
                self.message = msg

        self.choices = [Choice(Msg(content))]
        self.usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens}


class MockClient:
    def __init__(self, golden_map):
        self.golden = golden_map

    class chat:
        pass

    class _Inner:
        def __init__(self, golden):
            self._golden = golden

        async def create(self, model, messages, temperature=0.0):
            # emulate latency
            await asyncio.sleep(random.uniform(0.01, 0.05))
            # attempt to find snippet id in the last message text
            txt = ''
            if isinstance(messages, list) and messages:
                txt = messages[-1].get('content', '')
            # find an id token like j01..j10 inside text (not guaranteed). Fallback: pick random gold.
            sid = None
            for k in self._golden.keys():
                if k in txt:
                    sid = k
                    break
            if sid is None:
                sid = random.choice(list(self._golden.keys()))
            payload = self._golden[sid].copy()
            # ensure years is int or null
            if 'years_experience_required' in payload and payload['years_experience_required'] is not None:
                payload['years_experience_required'] = int(payload['years_experience_required'])
            content = json.dumps({'company': payload.get('company'), 'role': payload.get('role'), 'years_experience_required': payload.get('years_experience_required')}, ensure_ascii=False)
            return MockResp(content)

    def __post_init__(self):
        pass

    def bind(self):
        return MockClient._Inner(self.golden)


async def run_one(client_inner, strategy_name: str, snippet: dict, model: str, rates: dict) -> dict:
    messages = STRATEGIES[strategy_name](snippet.get('text', snippet.get('snippet', '')))
    t0 = time.perf_counter()
    resp = await client_inner.create(model=model, messages=messages, temperature=0.0)
    latency = time.perf_counter() - t0
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)
    parsed = parse_response(content)
    usage = getattr(resp, 'usage', {})
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    rates_model = rates.get(model, {'in': 0.0, 'out': 0.0})
    cost = prompt_tokens * rates_model['in'] + completion_tokens * rates_model['out']
    return {
        'strategy': strategy_name,
        'snippet_id': snippet.get('id'),
        'raw': content,
        'parsed': parsed,
        'cost_usd': cost,
        'latency_s': latency,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
    }


async def run_all(snippets, client_inner, model, rates):
    tasks = []
    for sname in STRATEGIES.keys():
        for snip in snippets:
            tasks.append(run_one(client_inner, sname, snip, model, rates))
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


def score_llm_judge(extracted: dict | None, gold: dict) -> int:
    # Simple deterministic judge for dry-run: map accuracy 0->1,1->2,2->3,3->4
    acc = score_accuracy(extracted, gold)
    return 1 + acc


def write_results_jsonl(results, path: Path):
    with path.open('w', encoding='utf8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    snippets, golden = load_data()
    mock = MockClient(golden)
    client_inner = mock._Inner(golden)
    MODEL = 'gpt-4o-mini'
    RATES = {'gpt-4o-mini': {'in': 0.0, 'out': 0.0}}

    results = asyncio.run(run_all(snippets, client_inner, MODEL, RATES))

    # scoring
    scored = []
    for r in results:
        gid = r['snippet_id']
        gold_entry = golden.get(gid, {})
        acc = score_accuracy(r.get('parsed'), gold_entry)
        r['accuracy'] = acc
        r['parse_success'] = 1 if r.get('parsed') is not None else 0
        # synchronous dry-run judge
        r['llm_judge_score'] = int(score_llm_judge(r.get('parsed'), gold_entry))
        scored.append(r)

    out_jsonl = RESULTS_DIR / 'mp1_results_dryrun.jsonl'
    write_results_jsonl(scored, out_jsonl)

    df = pd.DataFrame(scored)
    summary = df.groupby('strategy').agg({'accuracy': 'mean', 'parse_success': 'mean', 'llm_judge_score': 'mean', 'cost_usd': 'sum', 'latency_s': 'median'}).round(3)
    summary.columns = ['Accuracy (mean of 3)', 'Parse rate', 'Judge score', 'Total cost ($)', 'Latency p50 (s)']

    out_md = RESULTS_DIR / 'mp1_comparison_dryrun.md'
    out_md.write_text('# MP1 Dry-run comparison\n\n' + summary.to_markdown())

    print('Dry-run complete. Results written to:', out_jsonl)
    print(summary)


if __name__ == '__main__':
    main()
