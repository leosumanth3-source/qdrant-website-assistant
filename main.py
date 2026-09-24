import math
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(override=True)


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "dummy"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
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

INITIAL_TOP_K = 15

FINAL_TOP_K = 3

MAX_CONTEXT_CHARS = 4000

MAX_OUTPUT_TOKENS = 220

THINKING_LEVEL = "minimal"


# ============================================================
# VALIDATION
# ============================================================

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. "
        "Add it to your .env file."
    )
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Qdrant Website Assistant",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .source-box {
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #e3e3e3;
        margin-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Qdrant Website Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the publicly scraped Qdrant website.
    Answers are grounded in the retrieved website content.
    </div>
    """,
    unsafe_allow_html=True
)


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
    text: str
) -> float:

    combined = (
        f"{title} "
        f"{section} "
        f"{text}"
    ).lower()

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
    ]

    score = 0.0

    question_lower = question.lower()

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
    text: str
) -> float:

    question_words = tokenize(question)

    if not question_words:
        return 0.0

    title_words = tokenize(title)

    section_words = tokenize(section)

    text_words = tokenize(text)

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
        score / max(len(question_words), 1),
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
    text: str
) -> float:

    q = question.lower()
    t = title.lower()
    s = section.lower()
    p = page_type.lower()
    u = url.lower()
    body = text.lower()

    score = 0.0

    # --------------------------------------------------------
    # PRICING
    # --------------------------------------------------------

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
        "billing"
    }

    if any(term in q for term in pricing_terms):

        if "/pricing" in u:
            score += 0.30

        if "pricing" in t:
            score += 0.30

        if "pricing" in s:
            score += 0.25

        if "plan" in t or "plans" in t:
            score += 0.18

        if "tier" in t or "tiers" in t:
            score += 0.18

        if "plan" in s or "plans" in s:
            score += 0.15

        if "tier" in s or "tiers" in s:
            score += 0.15

        if "free" in q:

            if "free" in t:
                score += 0.20

            if "free" in s:
                score += 0.18

            if "free tier" in body:
                score += 0.15

        if "premium" in q:

            if "premium" in t:
                score += 0.20

            if "premium" in s:
                score += 0.18

            if "premium tier" in body:
                score += 0.15

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

    # --------------------------------------------------------
    # WHAT IS QDRANT?
    # --------------------------------------------------------

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
                "what is qdrant"
            ]
        ):
            score += 0.50

        if any(
            phrase in s
            for phrase in [
                "overview",
                "introduction",
                "what is qdrant"
            ]
        ):
            score += 0.30

        if p in {
            "documentation",
            "about",
            "homepage",
            "website"
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
                "hybrid search"
            ]
        ):
            score -= 0.20

    # --------------------------------------------------------
    # CAREERS
    # --------------------------------------------------------

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
        "employment"
    ]

    if any(term in q for term in career_terms):

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
                "about us"
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
                "leadership"
            ]
        ):
            score += 0.20

    # --------------------------------------------------------
    # CLOUD
    # --------------------------------------------------------

    if "cloud" in q:

        if p == "cloud":
            score += 0.30

        if "cloud" in t:
            score += 0.25

        if "cloud" in s:
            score += 0.20

        if "/cloud" in u:
            score += 0.15

    # --------------------------------------------------------
    # HYBRID SEARCH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TECHNICAL DOCUMENTATION
    # --------------------------------------------------------

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
        "shards"
    ]

    if any(
        word in q
        for word in technical_words
    ):

        if p == "documentation":
            score += 0.20

    return max(
        0.0,
        min(score, 1.0)
    )


# ============================================================
# BUILD RERANKER TEXT
# ============================================================

def build_candidate_text(
    payload: dict
) -> str:

    return (
        f"Title: {payload.get('title', '')}\n"
        f"Page type: {payload.get('page_type', '')}\n"
        f"Section: {payload.get('section', '')}\n\n"
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

    minimum = min(values)

    maximum = max(values)

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
# LIMIT CONTEXT
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
# CACHED RESOURCES
# ============================================================

@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


@st.cache_resource(
    show_spinner="Loading reranker..."
)
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )



@st.cache_resource
def load_qdrant():

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )


@st.cache_resource
def load_gemini():

    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

embedding_model = load_embedding_model()

reranker = load_reranker()

qdrant = load_qdrant()

gemini = load_gemini()


# ============================================================
# SEARCH FUNCTION
# ============================================================

def retrieve(
    question: str
):

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding_start = time.perf_counter()

    query_vector = embedding_model.encode(
        question,
        convert_to_numpy=True
    ).tolist()

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # --------------------------------------------------------
    # QDRANT
    # --------------------------------------------------------

    qdrant_start = time.perf_counter()

    results = qdrant.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        with_payload=True,

        limit=INITIAL_TOP_K

    ).points

    qdrant_time = (
        time.perf_counter()
        - qdrant_start
    )

    if not results:

        return {
            "results": [],
            "embedding_time": embedding_time,
            "qdrant_time": qdrant_time,
            "rerank_time": 0.0,
        }

    # --------------------------------------------------------
    # PREPARE CANDIDATES
    # --------------------------------------------------------

    candidate_pairs = []

    candidate_data = []

    for dense_rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload or {}

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

    # --------------------------------------------------------
    # CROSS ENCODER
    # --------------------------------------------------------

    rerank_start = time.perf_counter()

    rerank_scores = reranker.predict(
        candidate_pairs,
        show_progress_bar=False
    )

    rerank_time = (
        time.perf_counter()
        - rerank_start
    )

    # --------------------------------------------------------
    # RAW RERANK SCORES
    # --------------------------------------------------------

    for item, rerank_score in zip(
        candidate_data,
        rerank_scores
    ):

        item["rerank_score"] = float(
            rerank_score
        )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    dense_normalized = minmax_normalize(
        [
            item["dense_score"]
            for item in candidate_data
        ]
    )

    rerank_normalized = minmax_normalize(
        [
            item["rerank_score"]
            for item in candidate_data
        ]
    )

    lexical_normalized = minmax_normalize(
        [
            item["lexical_score"]
            for item in candidate_data
        ]
    )

    intent_normalized = minmax_normalize(
        [
            item["intent_score"]
            for item in candidate_data
        ]
    )

    # --------------------------------------------------------
    # HYBRID RANKING
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidate_data.sort(
        key=lambda item:
            item["final_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # SELECT TOP RESULTS
    # --------------------------------------------------------

    final_results = []

    seen_chunk_ids = set()

    seen_urls = {}

    for item in candidate_data:

        if len(final_results) >= FINAL_TOP_K:

            break

        payload = item["payload"]

        chunk_id = payload.get(
            "chunk_id",
            ""
        )

        url = payload.get(
            "url",
            ""
        )

        if chunk_id in seen_chunk_ids:

            continue

        page_count = seen_urls.get(
            url,
            0
        )

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

    return {
        "results": final_results,
        "embedding_time": embedding_time,
        "qdrant_time": qdrant_time,
        "rerank_time": rerank_time,
    }


# ============================================================
# GEMINI PROMPT
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
5. Use the supplied website content as the source of truth.
6. Combine the supplied chunks when they are relevant.
7. Give a concise and direct answer.
8. Do not mention the RAG system.
9. Do not mention embeddings or retrieval.
10. Do not provide code unless the user explicitly asks.

For pricing questions:
- distinguish Free, Standard, and Premium when the
  supplied content contains those distinctions.
- preserve the wording and differences from the source.
- never guess a missing price or feature.

For technical questions:
- explain only what is supported by the supplied content.

If the supplied content is insufficient, respond exactly:

The Qdrant website content in the knowledge base does not contain enough information to answer this question.
"""


# ============================================================
# GEMINI ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str
):

    user_content = (
        "Question:\n"
        + question
        + "\n\n"
        + "Qdrant website content:\n"
        + context
    )

    response = gemini.models.generate_content(

        model=GEMINI_MODEL,

        contents=user_content,

        config=types.GenerateContentConfig(

            system_instruction=
                SYSTEM_INSTRUCTION,

            max_output_tokens=
                MAX_OUTPUT_TOKENS,

            thinking_config=
                types.ThinkingConfig(
                    thinking_level=
                        THINKING_LEVEL
                )
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

    return answer.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System")

    st.write(
        f"**Vector DB:** {COLLECTION_NAME}"
    )

    st.write(
        f"**Embedding:** {EMBEDDING_MODEL}"
    )

    st.write(
        f"**Reranker:** {RERANKER_MODEL}"
    )

    st.write(
        f"**LLM:** {GEMINI_MODEL}"
    )

    st.divider()

    st.write(
        f"Initial retrieval: {INITIAL_TOP_K}"
    )

    st.write(
        f"Final chunks: {FINAL_TOP_K}"
    )

    st.write(
        f"Max context: {MAX_CONTEXT_CHARS} chars"
    )

    st.write(
        f"Max output: {MAX_OUTPUT_TOKENS} tokens"
    )

    st.divider()

    st.caption(
        "Knowledge source: publicly scraped qdrant.tech content."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Ask a question about Qdrant",
    placeholder=(
        "Example: Explain the free and premium plans"
    ),
    height=120
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "Ask Qdrant",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask_clicked:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()


    # ========================================================
    # TOTAL TIMER
    # ========================================================

    total_start = time.perf_counter()


    # ========================================================
    # RETRIEVAL
    # ========================================================

    try:

        with st.spinner(
            "Searching Qdrant knowledge base..."
        ):

            retrieval = retrieve(
                question
            )

    except Exception as error:

        st.error(
            f"Retrieval error: {error}"
        )

        st.stop()


    final_results = retrieval[
        "results"
    ]


    if not final_results:

        st.warning(
            "No relevant information was found."
        )

        st.stop()


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    with st.expander(
        "Retrieval details",
        expanded=False
    ):

        st.write(
            f"Candidates retrieved: "
            f"{INITIAL_TOP_K}"
        )

        st.write(
            f"Final chunks: "
            f"{len(final_results)}"
        )

        st.write(
            f"Embedding time: "
            f"{retrieval['embedding_time']:.3f}s"
        )

        st.write(
            f"Qdrant time: "
            f"{retrieval['qdrant_time']:.3f}s"
        )

        st.write(
            f"Reranker time: "
            f"{retrieval['rerank_time']:.3f}s"
        )


        st.markdown(
            "### Top retrieved chunks"
        )


        for index, item in enumerate(
            final_results,
            start=1
        ):

            payload = item["payload"]

            st.write(
                f"**{index}. "
                f"{payload.get('title', '')}**"
            )

            st.write(
                f"Section: "
                f"{payload.get('section', '')}"
            )

            st.write(
                f"Score: "
                f"{item['final_score']:.4f}"
            )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context_parts = []

    sources = []


    for item in final_results:

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


    context = "\n\n---\n\n".join(
        context_parts
    )


    context = limit_context(
        context
    )


    # ========================================================
    # CONTEXT SIZE
    # ========================================================

    st.caption(
        f"Context sent to Gemini: "
        f"{len(context):,} characters"
    )


    # ========================================================
    # GENERATE
    # ========================================================

    gemini_start = time.perf_counter()


    try:

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                context
            )

    except Exception as error:

        st.error(
            f"Gemini error: {error}"
        )

        st.stop()


    gemini_time = (
        time.perf_counter()
        - gemini_start
    )


    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start
    )


    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        "## Answer"
    )

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # TIMING
    # ========================================================

    st.markdown(
        "### Performance"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Embedding",
            f"{retrieval['embedding_time']:.3f}s"
        )


    with col2:

        st.metric(
            "Qdrant",
            f"{retrieval['qdrant_time']:.3f}s"
        )


    with col3:

        st.metric(
            "Gemini",
            f"{gemini_time:.3f}s"
        )


    with col4:

        st.metric(
            "Total",
            f"{total_time:.3f}s"
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.markdown(
        "## Sources"
    )


    for index, source in enumerate(
        sources,
        start=1
    ):

        with st.expander(
            f"{index}. {source['title']}"
        ):

            st.write(
                f"**Section:** "
                f"{source['section']}"
            )

            st.write(
                f"**Page type:** "
                f"{source['page_type']}"
            )

            st.write(
                f"**Retrieval score:** "
                f"{source['score']}"
            )

            st.markdown(
                f"[Open source page]({source['url']})"
            )


    # ========================================================
    # MODEL
    # ========================================================

    st.caption(
        f"Answer generated by: {GEMINI_MODEL}"
    )