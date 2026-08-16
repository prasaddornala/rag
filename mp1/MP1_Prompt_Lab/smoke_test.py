import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

DATA_DIR = Path(__file__).parent / 'data'

def load_first_snippet():
    lines = (DATA_DIR / 'job_snippets.jsonl').read_text().splitlines()
    for line in lines:
        if line.strip():
            return json.loads(line)
    return None

def prompt_structured(snippet_text: str) -> list[dict]:
    system = {
        'role': 'system',
        'content': (
            "You are an expert recruiter and a strict JSON formatter.\n"
            "Output MUST be a single JSON object with EXACT keys: company, role, years_experience_required.\n"
            "- company: string or null\n- role: string or null\n- years_experience_required: integer or null\n"
        )
    }
    user = {
        'role': 'user',
        'content': "Extract the schema values from the snippet below and return only the JSON object.\n\nSnippet:\n" + snippet_text
    }
    return [system, user]

async def main():
    snippet = load_first_snippet()
    if not snippet:
        print('No snippet found in data/job_snippets.jsonl')
        return

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
    assert OPENAI_API_KEY, 'OPENAI_API_KEY not set in environment'
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    MODEL = os.getenv('MODEL', 'gpt-4o-mini')

    messages = prompt_structured(snippet.get('text', snippet.get('snippet', '')))
    print('Sending a single structured prompt to model', MODEL)
    resp = await client.chat.completions.create(model=MODEL, messages=messages, temperature=0.0)
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)
    usage = getattr(resp, 'usage', {}) or (resp.get('usage') if isinstance(resp, dict) else {})
    print('--- RAW RESPONSE ---')
    print(content)
    print('--- USAGE ---')
    print(usage)

if __name__ == '__main__':
    asyncio.run(main())
