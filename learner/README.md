# MP2 Mini-RAG

This project answers questions about the supplied Sherlock Holmes corpus using
OpenAI embeddings, Qdrant retrieval, hybrid retrieval, cross-encoder reranking,
and GPT-4o-mini for grounded answers with citations.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Create `.env` beside this file using the Windows-compatible format:

   ```env
   OPENAI_API_KEY=your_openai_key
   QDRANT_URL=https://your-cluster-id.qdrant.io:6333
   QDRANT_API_KEY=your_qdrant_key
   ```

## Run

```powershell
python mp2_rag.py ingest
python mp2_rag.py validate
python mp2_rag.py ask
```

`validate` reports source-match results for the two supplied questions and, once
completed, the three learner-authored questions. A passing source match means a
retrieved citation includes the question's expected corpus file.