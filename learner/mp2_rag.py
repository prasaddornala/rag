"""MP2 Mini-RAG pipeline.

Run `python mp2_rag.py ingest` to build the collection, then use `ask` or
`validate` to query it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# These libraries read settings, create embeddings, search Qdrant, and answer questions.
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Load the private API settings from the local .env file.
load_dotenv()

# These values control the corpus, models, chunk size, and retrieval settings.
CORPUS_DIR = Path(__file__).parent / "corpus"
DATA_DIR = Path(__file__).parent / "data"
COLLECTION_NAME = "mp2_sherlock"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
RRF_K = 60
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

openai = OpenAI()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)
# The reranker is loaded only when the first question needs it.
# Keep the reranker ready to load later when the first question is asked.
reranker_model: CrossEncoder | None = None


# Read every story file and prepare it for the next RAG steps.
def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read all story files and keep their name, title, and complete text."""
    documents: list[dict[str, Any]] = []
    for file_path in sorted(corpus_dir.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not non_empty_lines:
            continue
        documents.append(
            {
                "source": file_path.name,
                "title": non_empty_lines[0],
                "text": text,
            }
        )
    return documents


# Split one story into small pieces so search can find the useful part.
def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Break one story into small overlapping pieces with source information."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", doc["text"])
        if paragraph.strip()
    ]
    chunks: list[dict[str, Any]] = []
    section = doc["title"]
    current_text = ""

    def add_chunk(text: str, chunk_section: str) -> None:
        cleaned_text = text.strip()
        if cleaned_text:
            chunks.append(
                {
                    "source": doc["source"],
                    "title": doc["title"],
                    "section": chunk_section,
                    "text": cleaned_text,
                }
            )

    def is_section_header(paragraph: str) -> bool:
        return (
            len(paragraph) <= 80
            and "\n" not in paragraph
            and not paragraph.endswith((".", "?", "!", ":", ";"))
        )

    for paragraph_index, paragraph in enumerate(paragraphs):
        if paragraph_index == 0:
            continue
        if is_section_header(paragraph):
            add_chunk(current_text, section)
            current_text = ""
            section = paragraph
            continue

        remaining = paragraph
        while remaining:
            available = TARGET_CHUNK_SIZE - len(current_text) - (2 if current_text else 0)
            if len(remaining) <= available:
                current_text = f"{current_text}\n\n{remaining}" if current_text else remaining
                remaining = ""
            else:
                if current_text:
                    add_chunk(current_text, section)
                    overlap = current_text[-CHUNK_OVERLAP:].lstrip()
                    remaining = f"{overlap}\n\n{remaining}"
                    current_text = ""
                    continue

                split_at = remaining.rfind(" ", 0, TARGET_CHUNK_SIZE)
                split_at = split_at if split_at > 0 else TARGET_CHUNK_SIZE
                add_chunk(remaining[:split_at], section)
                overlap = remaining[max(0, split_at - CHUNK_OVERLAP):split_at].lstrip()
                remaining = overlap + remaining[split_at:]

    add_chunk(current_text, section)
    return chunks


# Change text into number vectors that OpenAI can compare by meaning.
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert each text into a number vector so similar meanings can be searched."""
    if not texts:
        return []
    response = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


# Create the Qdrant place where story vectors and text details are stored.
def setup_collection() -> None:
    """Create a fresh Qdrant collection for storing the story vectors."""
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# Save all chunk vectors and their source information into Qdrant.
def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Create vectors for all chunks and save them with their text in Qdrant."""
    if not chunks:
        return
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


# Find story chunks whose meaning is closest to the user's question.
def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Find the closest story chunks to the question using vector similarity."""
    query_vector = embed_texts([query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    return [
        {**dict(point.payload or {}), "score": point.score}
        for point in response.points
    ]


# Combine meaning-based search with exact-word search for better results.
def hybrid_retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Combine meaning search and exact-word search before final reranking."""
    candidate_limit = max(k * 4, 12)
    dense_results = retrieve(query, k=candidate_limit)
    documents = load_corpus(CORPUS_DIR)
    all_chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document)
    ]
    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in all_chunks])
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_indices = sorted(
        range(len(all_chunks)), key=lambda index: bm25_scores[index], reverse=True
    )[:candidate_limit]
    bm25_results = [all_chunks[index] for index in bm25_indices]
    max_bm25_score = max(bm25_scores, default=0.0)

    fused: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rank, chunk in enumerate(dense_results, start=1):
        key = chunk_key(chunk)
        fused[key] = {**chunk, "rrf_score": 1 / (RRF_K + rank)}
    for rank, chunk in enumerate(bm25_results, start=1):
        key = chunk_key(chunk)
        if key not in fused:
            fused[key] = {**chunk, "rrf_score": 0.0}
        fused[key]["rrf_score"] += 1 / (RRF_K + rank)
        chunk_index = next(
            index for index, candidate in enumerate(all_chunks) if chunk_key(candidate) == key
        )
        fused[key]["bm25_score"] = float(bm25_scores[chunk_index])
        fused[key]["bm25_normalized"] = (
            float(bm25_scores[chunk_index] / max_bm25_score)
            if max_bm25_score
            else 0.0
        )

    return sorted(fused.values(), key=lambda chunk: chunk["rrf_score"], reverse=True)[
        :candidate_limit
    ]


# Reorder the selected chunks so the strongest evidence comes first.
def rerank(query: str, chunks: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    """Use a cross-encoder to put the most useful chunks at the top."""
    if not chunks:
        return []
    global reranker_model
    if reranker_model is None:
        reranker_model = CrossEncoder(RERANKER_MODEL)
    scores = reranker_model.predict(
        [
            (
                query,
                f"{chunk['title']} - {chunk['section']}\n{chunk['text']}",
            )
            for chunk in chunks
        ]
    )
    ranked_chunks = [
        {
            **chunk,
            "rerank_score": float(score),
            "combined_score": float(score) + 6 * chunk.get("bm25_normalized", 0.0),
        }
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    return sorted(ranked_chunks, key=lambda chunk: chunk["combined_score"], reverse=True)[:k]


# Break a sentence into simple lowercase words for BM25 search.
def tokenize(text: str) -> list[str]:
    """Split text into lowercase words for exact-word BM25 search."""
    return re.findall(r"\b\w+\b", text.lower())


# Build one key to identify the same chunk in different search results.
def chunk_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    """Create one ID so the same chunk is not counted twice during fusion."""
    return (chunk["source"], chunk["section"], chunk["text"])


SYSTEM_PROMPT = """You answer questions about the supplied Sherlock Holmes stories.
Use only the provided excerpts. If they do not contain the answer, say so plainly.
Cite the story title and section used for your answer."""


# Retrieve evidence, ask the model, and return the answer with citations.
def answer(question: str, k: int = 5) -> dict[str, Any]:
    """Find evidence, ask the language model, and return an answer with sources."""
    started_at = time.perf_counter()
    candidates = hybrid_retrieve(question, k=k)
    citations = rerank(question, candidates, k=k)
    context = "\n\n".join(
        f"[Source: {chunk['title']} - {chunk['section']}]\n{chunk['text']}"
        for chunk in citations
    )
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nExcerpts:\n{context}",
            },
        ],
    )
    return {
        "question": question,
        "answer": response.choices[0].message.content or "No answer returned.",
        "citations": [
            {
                "source": chunk["source"],
                "title": chunk["title"],
                "section": chunk["section"],
                "score": chunk.get("rerank_score"),
            }
            for chunk in citations
        ],
        "latency_ms": round((time.perf_counter() - started_at) * 1000),
    }


# Test questions and show whether the expected story was cited.
def validate_against(jsonl_path: Path) -> None:
    """Check questions from a JSONL file and print source and fact results."""
    questions = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"\nValidating {len(questions)} questions from {jsonl_path.name}...\n")

    hits = 0
    for question_data in questions:
        result = answer(question_data["question"], k=5)
        cited_sources = {citation["source"] for citation in result["citations"]}
        source_hit = question_data["expected_source"] in cited_sources
        answer_lower = result["answer"].lower()
        facts_hit = sum(
            fact.lower() in answer_lower
            for fact in question_data.get("expected_facts", [])
        )
        facts_total = len(question_data.get("expected_facts", []))

        print(f"{'PASS' if source_hit else 'FAIL'} {question_data['id']}")
        print(f"  Question: {question_data['question']}")
        print(f"  Cited: {', '.join(sorted(cited_sources))}")
        print(f"  Expected: {question_data['expected_source']}")
        print(f"  Facts matched: {facts_hit}/{facts_total}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")
        hits += int(source_hit)

    print(f"Source-match: {hits}/{len(questions)}")


# Run the complete loading, chunking, embedding, and storage process.
def cmd_ingest() -> None:
    """Load, split, embed, and store the complete story collection."""
    print("Loading corpus...")
    documents = load_corpus(CORPUS_DIR)
    all_chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
        print(f"  {document['source']}: {len(chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    setup_collection()
    ingest_chunks(all_chunks)
    print("Done. Try: python mp2_rag.py ask")


# Keep asking questions until the user gives an empty line or Ctrl-C.
def cmd_ask() -> None:
    """Keep accepting user questions until an empty line or Ctrl-C is given."""
    print("Mini-RAG over the Sherlock Holmes corpus. Empty line exits.\n")
    while True:
        try:
            question = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        result = answer(question, k=5)
        print(f"\n{result['answer']}\n")
        print("Sources:")
        for citation in result["citations"]:
            print(f"  - {citation['title']} - {citation['section']}")
        print(f"Latency: {result.get('latency_ms', '?')}ms\n")


# Run validation for both the supplied and learner-created questions.
def cmd_validate() -> None:
    """Validate both the supplied questions and the learner questions."""
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first_question = json.loads(learner_path.read_text(encoding="utf-8").splitlines()[0])
        if not first_question["question"].startswith("Replace this"):
            validate_against(learner_path)


# Read the command line and start the requested pipeline action.
def main() -> None:
    """Run the command selected by the user: ingest, ask, or validate."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    command = sys.argv[1]
    if command == "ingest":
        cmd_ingest()
    elif command == "ask":
        cmd_ask()
    elif command == "validate":
        cmd_validate()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()