import json
import math
import os
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from qdrant_client import QdrantClient

from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"

INDEX_FILE = STATIC_DIR / "index.html"


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

COLLECTION_NAME = "Qdrantdata"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

INITIAL_TOP_K = 20

FINAL_TOP_K = 3

MAX_CONTEXT_CHARS = 3000

MAX_OUTPUT_TOKENS = 180

THINKING_LEVEL = "minimal"


# ============================================================
# VALIDATION
# ============================================================

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is not set in .env"
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is not set in .env"
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in .env"
    )


# ============================================================
# FASTAPI
# ============================================================



# ============================================================
# REQUEST MODEL
# ============================================================

class Question(BaseModel):
    question: str


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "how",
    "does",
    "do",
    "can",
    "i",
    "we",
    "you",
    "with",
    "about",
    "tell",
    "me",
    "which",
    "when",
    "where",
    "why",
    "who",
    "was",
    "were",
    "will",
    "explain",
}


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text: str) -> set[str]:

    words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


# ============================================================
# PHRASE MATCHING
# ============================================================

def phrase_matches(
    question: str,
    title: str,
    section: str,
    text: str,
) -> float:

    combined = (
        f"{title} "
        f"{section} "
        f"{text}"
    ).lower()

    question_lower = question.lower()

    important_phrases = [
        "free tier",
        "standard tier",
        "premium tier",
        "hybrid search",
        "vector database",
        "vector search",
        "qdrant cloud",
        "semantic search",
        "sparse vectors",
        "dense vectors",
        "qdrant pricing",
        "features by tier",
    ]

    score = 0.0

    for phrase in important_phrases:

        if phrase in question_lower:

            if phrase in combined:
                score += 1.0

    return score


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(
    question: str,
    title: str,
    section: str,
    text: str,
) -> float:

    question_words = tokenize(
        question
    )

    if not question_words:
        return 0.0

    title_words = tokenize(
        title
    )

    section_words = tokenize(
        section
    )

    text_words = tokenize(
        text
    )

    title_overlap = len(
        question_words & title_words
    )

    section_overlap = len(
        question_words & section_words
    )

    text_overlap = len(
        question_words & text_words
    )

    phrase_score = phrase_matches(
        question,
        title,
        section,
        text
    )

    score = (
        title_overlap * 0.45
        + section_overlap * 0.30
        + text_overlap * 0.10
        + phrase_score * 0.15
    )

    return min(
        score / max(
            len(question_words),
            1
        ),
        1.0
    )


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(
    question: str,
    title: str,
    section: str,
    page_type: str,
    url: str,
    text: str,
) -> float:

    q = question.lower()
    t = title.lower()
    s = section.lower()
    p = page_type.lower()
    u = url.lower()
    body = text.lower()

    score = 0.0


    # ========================================================
    # PRICING
    # ========================================================

    pricing_terms = {
        "price",
        "pricing",
        "plan",
        "plans",
        "tier",
        "tiers",
        "free",
        "premium",
        "standard",
        "cost",
        "billing",
    }

    is_pricing_query = any(
        term in q
        for term in pricing_terms
    )

    if is_pricing_query:

        if "/pricing" in u:
            score += 0.30

        if "pricing" in t:
            score += 0.30

        if "pricing" in s:
            score += 0.25

        if (
            "plan" in t
            or "plans" in t
        ):
            score += 0.18

        if (
            "tier" in t
            or "tiers" in t
        ):
            score += 0.18

        if (
            "plan" in s
            or "plans" in s
        ):
            score += 0.15

        if (
            "tier" in s
            or "tiers" in s
        ):
            score += 0.15

        # Free
        if "free" in q:

            if "free" in t:
                score += 0.20

            if "free" in s:
                score += 0.18

            if "free tier" in body:
                score += 0.15

        # Premium
        if "premium" in q:

            if "premium" in t:
                score += 0.20

            if "premium" in s:
                score += 0.18

            if "premium tier" in body:
                score += 0.15

        # Standard
        if "standard" in q:

            if "standard" in t:
                score += 0.16

            if "standard" in s:
                score += 0.14

            if "standard tier" in body:
                score += 0.12

        if p == "pricing":
            score += 0.30

        if "billing" in t:
            score += 0.08

        if "billing" in s:
            score += 0.06


    # ========================================================
    # WHAT IS QDRANT?
    # ========================================================

    if (
        "what is qdrant" in q
        or "what does qdrant do" in q
        or "explain qdrant" in q
    ):

        if any(
            phrase in t
            for phrase in [
                "overview",
                "introduction",
                "vector search engine",
                "what is qdrant",
            ]
        ):
            score += 0.50

        if any(
            phrase in s
            for phrase in [
                "overview",
                "introduction",
                "what is qdrant",
            ]
        ):
            score += 0.30

        if p in {
            "documentation",
            "about",
            "homepage",
            "website",
        }:
            score += 0.10

        if any(
            word in t
            for word in [
                "pricing",
                "cloud",
                "edge",
                "web ui",
                "stars",
                "academy",
                "certification",
                "release",
                "hybrid search",
            ]
        ):
            score -= 0.20


    # ========================================================
    # CAREERS
    # ========================================================

    career_terms = [
        "career",
        "careers",
        "job",
        "jobs",
        "hiring",
        "employee",
        "employees",
        "working at",
        "work at",
        "open role",
        "open roles",
        "position",
        "positions",
        "employment",
    ]

    if any(
        term in q
        for term in career_terms
    ):

        if p == "careers":
            score += 0.40

        if p == "about":
            score += 0.20

        if any(
            word in t
            for word in [
                "career",
                "careers",
                "jobs",
                "join our team",
                "about us",
            ]
        ):
            score += 0.30

        if any(
            word in s
            for word in [
                "career",
                "careers",
                "join",
                "team",
                "roles",
                "leadership",
            ]
        ):
            score += 0.20


    # ========================================================
    # CLOUD
    # ========================================================

    if "cloud" in q:

        if p == "cloud":
            score += 0.30

        if "cloud" in t:
            score += 0.25

        if "cloud" in s:
            score += 0.20

        if "/cloud" in u:
            score += 0.15


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    if (
        "hybrid search" in q
        or "hybrid retrieval" in q
    ):

        if "hybrid" in t:
            score += 0.40

        if "hybrid" in s:
            score += 0.30

        if "hybrid search" in body:
            score += 0.15

        if "hybrid" in u:
            score += 0.10


    # ========================================================
    # TECHNICAL DOCUMENTATION
    # ========================================================

    technical_words = [
        "collection",
        "payload",
        "vector",
        "vectors",
        "filter",
        "filters",
        "search",
        "hnsw",
        "quantization",
        "embedding",
        "embeddings",
        "reranking",
        "api",
        "client",
        "shard",
        "shards",
    ]

    if any(
        word in q
        for word in technical_words
    ):

        if p == "documentation":
            score += 0.20


    return max(
        0.0,
        min(
            score,
            1.0
        )
    )


# ============================================================
# CANDIDATE TEXT
# ============================================================

def build_candidate_text(
    payload: dict[str, Any]
) -> str:

    return (
        f"Title: "
        f"{payload.get('title', '')}\n"

        f"Page type: "
        f"{payload.get('page_type', '')}\n"

        f"Section: "
        f"{payload.get('section', '')}\n\n"

        f"{payload.get('text', '')}"
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def minmax_normalize(
    values: list[float]
) -> list[float]:

    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if math.isclose(
        minimum,
        maximum
    ):
        return [
            0.5
            for _ in values
        ]

    return [
        (value - minimum)
        / (maximum - minimum)
        for value in values
    ]


# ============================================================
# CONTEXT LIMIT
# ============================================================

def limit_context(
    context: str
) -> str:

    if len(context) <= MAX_CONTEXT_CHARS:
        return context

    shortened = context[
        :MAX_CONTEXT_CHARS
    ]

    shortened = shortened.rsplit(
        " ",
        1
    )[0]

    return (
        shortened
        + "\n\n[Context truncated]"
    )


# ============================================================
# LAZY LOAD EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model():

    from sentence_transformers import (
        SentenceTransformer
    )

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================
# LAZY LOAD RERANKER
# ============================================================

@lru_cache(maxsize=1)
def load_reranker():

    from sentence_transformers import (
        CrossEncoder
    )

    print(
        "\nLoading reranker..."
    )

    model = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "Reranker loaded."
    )

    return model


# ============================================================
# QDRANT CLIENT
# ============================================================

@lru_cache(maxsize=1)
def load_qdrant():

    print(
        "\nConnecting to Qdrant Cloud..."
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
        prefer_grpc=True,
    )

    print(
        "Qdrant client ready."
    )

    return client


# ============================================================
# GEMINI CLIENT
# ============================================================

@lru_cache(maxsize=1)
def load_gemini():

    print(
        "\nConnecting to Gemini..."
    )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    print(
        "Gemini client ready."
    )

    return client


# ============================================================
# STARTUP RESOURCE PRELOAD
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n" + "=" * 90)
    print("STARTING QDRANT WEBSITE ASSISTANT")
    print("=" * 90)

    startup_start = time.perf_counter()

    try:

        print("\nPreloading embedding model...")
        load_embedding_model()

        print("\nPreloading reranker...")
        load_reranker()

        print("\nConnecting to Qdrant Cloud...")
        load_qdrant()

        print("\nInitializing Gemini...")
        load_gemini()

        startup_time = time.perf_counter() - startup_start

        print(
            f"\nStartup resources ready in "
            f"{startup_time:.3f}s"
        )
        print("=" * 90)

        yield

    finally:
        print("\nApplication shutdown complete.")


app = FastAPI(
    title="Qdrant Website Assistant API",
    description="RAG backend for the Qdrant website",
    version="8.1.0",
    lifespan=lifespan
)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question: str
) -> dict[str, Any]:

    retrieval_start = time.perf_counter()


    # ========================================================
    # LOAD RESOURCES LAZILY
    # ========================================================

    embedding_model = (
        load_embedding_model()
    )

    reranker = load_reranker()

    qdrant = load_qdrant()


    # ========================================================
    # EMBEDDING
    # ========================================================

    embedding_start = time.perf_counter()


    query_vector = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()


    embedding_time = (
        time.perf_counter()
        - embedding_start
    )


    # ========================================================
    # QDRANT SEARCH
    # ========================================================

    qdrant_start = time.perf_counter()


    try:

        results = qdrant.query_points(

            collection_name=
                COLLECTION_NAME,

            query=
                query_vector,

            with_payload=[
                "chunk_id",
                "title",
                "page_type",
                "section",
                "text",
                "url",
            ],

            limit=
                INITIAL_TOP_K
        ).points

    except Exception as error:

        raise RuntimeError(
            "Qdrant search failed: "
            f"{error}"
        )


    qdrant_time = (
        time.perf_counter()
        - qdrant_start
    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        return {

            "results": [],

            "embedding_time":
                embedding_time,

            "qdrant_time":
                qdrant_time,

            "rerank_time":
                0.0,

            "retrieval_time":
                time.perf_counter()
                - retrieval_start,
        }


    # ========================================================
    # PREPARE RERANKER INPUT
    # ========================================================

    candidate_pairs = []

    candidate_data = []


    for dense_rank, result in enumerate(
        results,
        start=1
    ):

        payload = (
            result.payload
            or {}
        )


        title = payload.get(
            "title",
            ""
        )

        section = payload.get(
            "section",
            ""
        )

        text = payload.get(
            "text",
            ""
        )

        page_type = payload.get(
            "page_type",
            "website"
        )

        url = payload.get(
            "url",
            ""
        )


        candidate_pairs.append(
            [
                question,
                build_candidate_text(
                    payload
                )
            ]
        )


        candidate_data.append({

            "payload":
                payload,

            "dense_score":
                float(
                    getattr(
                        result,
                        "score",
                        0.0
                    )
                ),

            "dense_rank":
                dense_rank,

            "lexical_score":
                lexical_score(
                    question,
                    title,
                    section,
                    text
                ),

            "intent_score":
                intent_score(
                    question,
                    title,
                    section,
                    page_type,
                    url,
                    text
                )
        })


    # ========================================================
    # CROSS ENCODER RERANKING
    # ========================================================

    rerank_start = time.perf_counter()


    try:

        rerank_scores = (
            reranker.predict(
                candidate_pairs,
                batch_size=8,
                show_progress_bar=False
            )
        )

    except Exception as error:

        raise RuntimeError(
            "Reranking failed: "
            f"{error}"
        )


    rerank_time = (
        time.perf_counter()
        - rerank_start
    )


    # ========================================================
    # STORE RERANK SCORES
    # ========================================================

    for item, rerank_score in zip(
        candidate_data,
        rerank_scores
    ):

        item["rerank_score"] = (
            float(
                rerank_score
            )
        )


    # ========================================================
    # NORMALIZE SIGNALS
    # ========================================================

    dense_normalized = (
        minmax_normalize(
            [
                item["dense_score"]
                for item in candidate_data
            ]
        )
    )


    rerank_normalized = (
        minmax_normalize(
            [
                item["rerank_score"]
                for item in candidate_data
            ]
        )
    )


    lexical_normalized = (
        minmax_normalize(
            [
                item["lexical_score"]
                for item in candidate_data
            ]
        )
    )


    intent_normalized = (
        minmax_normalize(
            [
                item["intent_score"]
                for item in candidate_data
            ]
        )
    )


    # ========================================================
    # HYBRID RANKING
    # ========================================================
    #
    # CrossEncoder: 45%
    # Dense:        20%
    # Lexical:      15%
    # Intent:       20%
    #
    # All values are normalized first.
    # ========================================================

    for index, item in enumerate(
        candidate_data
    ):

        item["dense_normalized"] = (
            dense_normalized[index]
        )

        item["rerank_normalized"] = (
            rerank_normalized[index]
        )

        item["lexical_normalized"] = (
            lexical_normalized[index]
        )

        item["intent_normalized"] = (
            intent_normalized[index]
        )


        item["final_score"] = (

            item["rerank_normalized"]
            * 0.45

            +

            item["dense_normalized"]
            * 0.20

            +

            item["lexical_normalized"]
            * 0.15

            +

            item["intent_normalized"]
            * 0.20
        )


    # ========================================================
    # SORT
    # ========================================================

    candidate_data.sort(
        key=lambda item:
            item["final_score"],
        reverse=True
    )


    # ========================================================
    # FINAL RESULT SELECTION
    # ========================================================

    final_results = []

    seen_chunk_ids = set()

    seen_urls: dict[str, int] = {}


    for item in candidate_data:

        if len(final_results) >= FINAL_TOP_K:
            break


        payload = item["payload"]


        chunk_id = str(
            payload.get(
                "chunk_id",
                ""
            )
        )

        url = str(
            payload.get(
                "url",
                ""
            )
        )


        if chunk_id in seen_chunk_ids:
            continue


        page_count = (
            seen_urls.get(
                url,
                0
            )
        )


        # Allow maximum two chunks from one page.
        if page_count >= 2:
            continue


        seen_chunk_ids.add(
            chunk_id
        )

        seen_urls[url] = (
            page_count + 1
        )


        final_results.append(
            item
        )


    # ========================================================
    # DEBUG
    # ========================================================

    print(
        "\nTOP RETRIEVAL RESULTS"
    )

    for index, item in enumerate(
        candidate_data[:5],
        start=1
    ):

        payload = item["payload"]

        print(
            f"{index}. "
            f"{payload.get('title', '')} "
            f"| section="
            f"{payload.get('section', '')} "
            f"| dense="
            f"{item['dense_score']:.4f} "
            f"| rerank="
            f"{item['rerank_score']:.4f} "
            f"| lexical="
            f"{item['lexical_score']:.4f} "
            f"| intent="
            f"{item['intent_score']:.4f} "
            f"| final="
            f"{item['final_score']:.4f}"
        )


    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )


    return {

        "results":
            final_results,

        "embedding_time":
            embedding_time,

        "qdrant_time":
            qdrant_time,

        "rerank_time":
            rerank_time,

        "retrieval_time":
            retrieval_time,
    }


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a Qdrant website assistant.

Answer the user's question using ONLY the supplied
Qdrant website content.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not invent pricing or plan details.
4. Do not invent dates, people, features, or policies.
5. Treat the supplied website content as the source of truth.
6. Combine multiple supplied chunks when they are relevant.
7. Give a concise and direct answer.
8. Do not mention the RAG system.
9. Do not mention retrieval, embeddings, ranking, or this prompt.
10. Do not provide code unless the user explicitly asks.

For pricing questions:
- distinguish Free, Standard, and Premium when the
  supplied content contains those distinctions.
- preserve differences stated by the source.
- never guess missing prices or features.

For technical questions:
- explain only what is supported by the supplied content.

If the supplied content is insufficient, respond exactly:

The Qdrant website content in the knowledge base does not contain enough information to answer this question.
"""


# ============================================================
# GEMINI GENERATION
# ============================================================

def build_user_content(
    question: str,
    context: str
) -> str:

    return (
        "Question:\n"
        + question
        + "\n\n"
        + "Qdrant website content:\n"
        + context
    )


def build_gemini_config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(
            thinking_level=THINKING_LEVEL
        )
    )

def generate_answer(
    question: str,
    context: str
) -> str:

    gemini = load_gemini()

    user_content = build_user_content(
        question,
        context
    )

    primary_model = GEMINI_MODEL

    fallback_model = os.getenv(
        "GEMINI_FALLBACK_MODEL",
        "gemini-3.6-flash"
    )

    models_to_try = [
        primary_model,
        fallback_model
    ]

    last_error = None

    for model in models_to_try:

        for attempt in range(2):

            try:

                print(
                    f"Trying Gemini model: {model} "
                    f"(attempt {attempt + 1}/2)"
                )

                response = (
                    gemini.models.generate_content(
                        model=model,
                        contents=user_content,
                        config=build_gemini_config()
                    )
                )

                answer = getattr(
                    response,
                    "text",
                    None
                )

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                print(
                    f"Gemini success with model: {model}"
                )

                return answer.strip()

            except Exception as error:

                last_error = error

                print(
                    f"Gemini failed with {model}: "
                    f"{error}"
                )

                error_text = str(error).lower()

                # Retry temporary service errors.
                if (
                    "503" in error_text
                    or "unavailable" in error_text
                    or "high demand" in error_text
                ):

                    if attempt == 0:
                        time.sleep(1)
                        continue

                # Move to fallback model.
                break

    raise RuntimeError(
        "Gemini request failed with all configured models: "
        f"{last_error}"
    )


def stream_answer(
    question: str,
    context: str
):

    gemini = load_gemini()

    user_content = build_user_content(
        question,
        context
    )

    try:

        return gemini.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=user_content,
            config=build_gemini_config()
        )

    except Exception as error:

        raise RuntimeError(
            "Gemini streaming request failed: "
            f"{error}"
        )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:

    context_parts = []

    sources = []


    for item in results:

        payload = item["payload"]


        title = payload.get(
            "title",
            ""
        )

        page_type = payload.get(
            "page_type",
            ""
        )

        section = payload.get(
            "section",
            ""
        )

        text = payload.get(
            "text",
            ""
        )

        url = payload.get(
            "url",
            ""
        )


        context_parts.append(

            f"Title: {title}\n"
            f"Page type: {page_type}\n"
            f"Section: {section}\n\n"
            f"Content:\n{text}"
        )


        sources.append({

            "title":
                title,

            "page_type":
                page_type,

            "section":
                section,

            "url":
                url,

            "score":
                round(
                    item["final_score"],
                    4
                )
        })


    context = (
        "\n\n---\n\n".join(
            context_parts
        )
    )


    context = limit_context(
        context
    )


    return (
        context,
        sources
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    if not INDEX_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "static/index.html "
                "was not found."
            )
        )


    return FileResponse(
        INDEX_FILE
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        qdrant = load_qdrant()


        info = qdrant.get_collection(
            collection_name=
                COLLECTION_NAME
        )


        return {

            "status":
                "ok",

            "collection":
                COLLECTION_NAME,

            "points":
                info.points_count,

            "embedding_model":
                EMBEDDING_MODEL,

            "reranker_model":
                RERANKER_MODEL,

            "gemini_model":
                GEMINI_MODEL,

            "initial_top_k":
                INITIAL_TOP_K,

            "final_top_k":
                FINAL_TOP_K,

            "max_context_chars":
                MAX_CONTEXT_CHARS
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# ASK HELPERS
# ============================================================

INSUFFICIENT_CONTENT_MESSAGE = (
    "The Qdrant website content in the knowledge base does not contain "
    "enough information to answer this question."
)


def prepare_retrieval(
    request: Question
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    retrieval = retrieve(question)

    return question, retrieval


def json_line(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False
    ) + "\n"


# ============================================================
# ASK - NORMAL JSON RESPONSE
# ============================================================

@app.post("/ask")
def ask(
    request: Question
):

    request_start = time.perf_counter()

    try:
        question, retrieval = prepare_retrieval(request)
    except HTTPException:
        raise
    except Exception as error:
        print("\nRETRIEVAL ERROR:", error)
        raise HTTPException(status_code=500, detail=str(error))

    final_results = retrieval["results"]

    if not final_results:
        total_time = time.perf_counter() - request_start
        return {
            "question": question,
            "answer": INSUFFICIENT_CONTENT_MESSAGE,
            "retrieved_chunks": 0,
            "model_used": None,
            "sources": [],
            "timing": {
                "embedding_seconds": round(retrieval["embedding_time"], 3),
                "qdrant_seconds": round(retrieval["qdrant_time"], 3),
                "reranker_seconds": round(retrieval["rerank_time"], 3),
                "gemini_seconds": 0.0,
                "total_seconds": round(total_time, 3),
                "context_characters": 0
            }
        }

    context, sources = build_context(final_results)
    print(f"Context sent to Gemini: {len(context)} characters")

    gemini_start = time.perf_counter()

    try:
        answer = generate_answer(question, context)
    except Exception as error:
        gemini_time = time.perf_counter() - gemini_start
        print("\nGEMINI ERROR:", error)
        raise HTTPException(status_code=500, detail=str(error))

    gemini_time = time.perf_counter() - gemini_start
    total_time = time.perf_counter() - request_start

    print(f"Embedding time: {retrieval['embedding_time']:.3f}s")
    print(f"Qdrant time: {retrieval['qdrant_time']:.3f}s")
    print(f"Reranker time: {retrieval['rerank_time']:.3f}s")
    print(f"Gemini time: {gemini_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print("=" * 90)

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": len(final_results),
        "model_used": GEMINI_MODEL,
        "sources": sources,
        "timing": {
            "embedding_seconds": round(retrieval["embedding_time"], 3),
            "qdrant_seconds": round(retrieval["qdrant_time"], 3),
            "reranker_seconds": round(retrieval["rerank_time"], 3),
            "gemini_seconds": round(gemini_time, 3),
            "total_seconds": round(total_time, 3),
            "context_characters": len(context)
        }
    }


# ============================================================
# ASK/STREAM - NDJSON STREAMING RESPONSE
# ============================================================

@app.post("/ask/stream")
def ask_stream(
    request: Question
):

    def event_generator():

        request_start = time.perf_counter()

        try:
            question, retrieval = prepare_retrieval(request)
        except HTTPException as error:
            yield json_line({
                "type": "error",
                "message": error.detail
            })
            return
        except Exception as error:
            print("\nRETRIEVAL ERROR:", error)
            yield json_line({
                "type": "error",
                "message": str(error)
            })
            return

        final_results = retrieval["results"]

        if not final_results:
            total_time = time.perf_counter() - request_start

            yield json_line({
                "type": "sources",
                "sources": []
            })

            yield json_line({
                "type": "token",
                "text": INSUFFICIENT_CONTENT_MESSAGE
            })

            yield json_line({
                "type": "done",
                "model_used": None,
                "timing": {
                    "embedding_seconds": round(retrieval["embedding_time"], 3),
                    "qdrant_seconds": round(retrieval["qdrant_time"], 3),
                    "reranker_seconds": round(retrieval["rerank_time"], 3),
                    "gemini_seconds": 0.0,
                    "total_seconds": round(total_time, 3),
                    "context_characters": 0
                }
            })
            return

        context, sources = build_context(final_results)
        print(f"Context sent to Gemini: {len(context)} characters")

        yield json_line({
            "type": "sources",
            "sources": sources
        })

        gemini_start = time.perf_counter()

        try:
            response_stream = stream_answer(question, context)

            for chunk in response_stream:
                chunk_text = getattr(chunk, "text", "") or ""

                if not chunk_text:
                    continue

                yield json_line({
                    "type": "token",
                    "text": chunk_text
                })

        except Exception as error:
            print("\nGEMINI STREAM ERROR:", error)
            yield json_line({
                "type": "error",
                "message": str(error)
            })
            return

        gemini_time = time.perf_counter() - gemini_start
        total_time = time.perf_counter() - request_start

        print(f"Embedding time: {retrieval['embedding_time']:.3f}s")
        print(f"Qdrant time: {retrieval['qdrant_time']:.3f}s")
        print(f"Reranker time: {retrieval['rerank_time']:.3f}s")
        print(f"Gemini time: {gemini_time:.3f}s")
        print(f"Total time: {total_time:.3f}s")
        print("=" * 90)

        yield json_line({
            "type": "done",
            "model_used": GEMINI_MODEL,
            "retrieved_chunks": len(final_results),
            "timing": {
                "embedding_seconds": round(retrieval["embedding_time"], 3),
                "qdrant_seconds": round(retrieval["qdrant_time"], 3),
                "reranker_seconds": round(retrieval["rerank_time"], 3),
                "gemini_seconds": round(gemini_time, 3),
                "total_seconds": round(total_time, 3),
                "context_characters": len(context)
            }
        })

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )
