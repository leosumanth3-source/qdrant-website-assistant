import re

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "Qdrantdata"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INITIAL_TOP_K = 30

FINAL_TOP_K = 5

SAME_PAGE_CHUNKS = 2


# ============================================================
# CONNECT TO QDRANT
# ============================================================

print("Connecting to Qdrant...")

client = QdrantClient(
    url=QDRANT_URL
)

print("Connected.")


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


print("Loading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker loaded.")


# ============================================================
# TEXT HELPERS
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase words.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
            text.lower()
        )
    )


def lexical_score(
    question,
    title,
    section,
    text
):
    """
    Small lexical relevance score.

    Title and section receive more importance than body text.
    """

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


    # Weighted score
    score = (
        title_overlap * 3.0
        + section_overlap * 2.0
        + text_overlap * 0.25
    )


    return score


# ============================================================
# INTENT BOOST
# ============================================================

def intent_boost(
    question,
    title,
    page_type
):
    """
    Give a small boost to pages whose type/title matches
    the apparent question intent.
    """

    q = question.lower()
    t = title.lower()
    p = page_type.lower()


    boost = 0.0


    # --------------------------------------------------------
    # Definition / overview questions
    # --------------------------------------------------------

    definition_patterns = [
        "what is",
        "what are",
        "what does",
        "explain",
        "meaning of",
        "overview of",
        "introduction to",
    ]


    is_definition_question = any(
        pattern in q
        for pattern in definition_patterns
    )


    if is_definition_question:

        if any(
            word in t
            for word in [
                "overview",
                "introduction",
                "what is",
                "getting started",
            ]
        ):

            boost += 2.5


        if p in {
            "documentation",
            "about",
            "homepage",
        }:

            boost += 0.5


    # --------------------------------------------------------
    # Pricing questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "price",
            "pricing",
            "cost",
            "plan",
            "plans",
            "free tier",
            "premium",
        ]
    ):

        if p == "pricing":
            boost += 4.0

        if "pricing" in t:
            boost += 3.0


    # --------------------------------------------------------
    # Career questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "career",
            "job",
            "jobs",
            "hiring",
            "employee",
            "work at",
            "working at",
        ]
    ):

        if p in {
            "careers",
            "about",
        }:

            boost += 3.0

        if any(
            word in t
            for word in [
                "career",
                "careers",
                "jobs",
                "about us",
            ]
        ):

            boost += 2.0


    # --------------------------------------------------------
    # Cloud questions
    # --------------------------------------------------------

    if "cloud" in q:

        if p == "cloud":
            boost += 3.0

        if "cloud" in t:
            boost += 2.0


    # --------------------------------------------------------
    # Search questions
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "search",
            "retrieval",
            "vector search",
            "hybrid search",
            "semantic search",
        ]
    ):

        if "search" in t:
            boost += 2.0

        if "search" in p:
            boost += 1.0


    return boost


# ============================================================
# GET SAME PAGE CHUNKS
# ============================================================

def get_same_page_chunks(
    page_url,
    exclude_chunk_id
):
    """
    Retrieve other chunks belonging to the same page.
    """

    if not page_url:
        return []


    page_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="url",
                match=models.MatchValue(
                    value=page_url
                )
            )
        ]
    )


    try:

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=page_filter,
            with_payload=True,
            limit=100,
        )

    except Exception as error:

        print(
            "Same-page retrieval failed:",
            error
        )

        return []


    chunks = []


    for point in points:

        payload = point.payload or {}

        chunk_id = payload.get(
            "chunk_id",
            ""
        )


        if chunk_id == exclude_chunk_id:
            continue


        chunks.append(
            payload
        )


    return chunks


# ============================================================
# FIND NEIGHBORING CHUNKS
# ============================================================

def find_neighbors(
    anchor_payload,
    same_page_chunks
):
    """
    Find nearby chunk numbers around the retrieved chunk.
    """

    anchor_number = anchor_payload.get(
        "chunk_number"
    )


    if not isinstance(
        anchor_number,
        int
    ):
        return []


    neighbors = []


    for payload in same_page_chunks:

        chunk_number = payload.get(
            "chunk_number"
        )


        if not isinstance(
            chunk_number,
            int
        ):
            continue


        distance = abs(
            chunk_number - anchor_number
        )


        if (
            1 <= distance <= SAME_PAGE_CHUNKS
        ):

            neighbors.append(
                (
                    distance,
                    payload
                )
            )


    neighbors.sort(
        key=lambda item: item[0]
    )


    return [
        payload
        for _, payload in neighbors
    ]


# ============================================================
# QUESTION
# ============================================================

question = input(
    "\nAsk a question about the Qdrant website: "
).strip()


if not question:

    print(
        "Question cannot be empty."
    )

    raise SystemExit


# ============================================================
# QUERY EMBEDDING
# ============================================================

print(
    "\nCreating query embedding..."
)

query_vector = embedding_model.encode(
    question,
    convert_to_numpy=True
).tolist()


# ============================================================
# DENSE RETRIEVAL
# ============================================================

print(
    "Searching Qdrantdata..."
)


results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    with_payload=True,
    limit=INITIAL_TOP_K,
).points


if not results:

    print(
        "No results found."
    )

    raise SystemExit


print(
    f"Retrieved {len(results)} initial candidates."
)


# ============================================================
# SCORE CANDIDATES
# ============================================================

candidate_data = []


for result in results:

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


    # --------------------------------------------------------
    # Lexical score
    # --------------------------------------------------------

    lexical = lexical_score(
        question,
        title,
        section,
        text
    )


    # --------------------------------------------------------
    # Intent score
    # --------------------------------------------------------

    intent = intent_boost(
        question,
        title,
        page_type
    )


    candidate_text = (
        f"Title: {title}\n"
        f"Page type: {page_type}\n"
        f"Section: {section}\n\n"
        f"{text}"
    )


    candidate_data.append({

        "result": result,

        "payload": payload,

        "candidate_text": candidate_text,

        "lexical_score": lexical,

        "intent_score": intent,

    })


# ============================================================
# RERANK
# ============================================================

print(
    "Reranking candidates..."
)


pairs = [
    [
        question,
        item["candidate_text"]
    ]
    for item in candidate_data
]


rerank_scores = reranker.predict(
    pairs
)


# ============================================================
# FINAL COMBINED SCORE
# ============================================================

for item, rerank_score in zip(
    candidate_data,
    rerank_scores
):

    item["rerank_score"] = float(
        rerank_score
    )


    # Reranker is the primary signal.
    # Lexical and intent scores are supporting signals.
    item["final_score"] = (
        item["rerank_score"]
        + item["lexical_score"] * 0.20
        + item["intent_score"]
    )


candidate_data.sort(
    key=lambda item: item["final_score"],
    reverse=True
)


# ============================================================
# SAME-PAGE CONTEXT EXPANSION
# ============================================================

expanded_results = []

seen_chunk_ids = set()


for item in candidate_data:

    if len(expanded_results) >= FINAL_TOP_K:
        break


    result = item["result"]

    payload = item["payload"]

    chunk_id = payload.get(
        "chunk_id",
        ""
    )


    if chunk_id in seen_chunk_ids:
        continue


    seen_chunk_ids.add(
        chunk_id
    )


    expanded_results.append({
        "type": "primary",
        "payload": payload,
        "vector_score": result.score,
        "rerank_score": item["rerank_score"],
        "lexical_score": item["lexical_score"],
        "intent_score": item["intent_score"],
        "final_score": item["final_score"],
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("FINAL RETRIEVAL RESULTS")
print("=" * 90)


for index, item in enumerate(
    expanded_results,
    start=1
):

    payload = item["payload"]


    print("\n" + "-" * 90)

    print(
        f"RESULT {index}"
    )

    print("-" * 90)

    print(
        "Final score:",
        round(
            item["final_score"],
            4
        )
    )

    print(
        "Reranker score:",
        round(
            item["rerank_score"],
            4
        )
    )

    print(
        "Vector score:",
        round(
            item["vector_score"],
            4
        )
    )

    print(
        "Lexical score:",
        round(
            item["lexical_score"],
            4
        )
    )

    print(
        "Intent boost:",
        round(
            item["intent_score"],
            4
        )
    )

    print(
        "Title:",
        payload.get(
            "title",
            ""
        )
    )

    print(
        "Page type:",
        payload.get(
            "page_type",
            ""
        )
    )

    print(
        "Section:",
        payload.get(
            "section",
            ""
        )
    )

    print(
        "Chunk:",
        payload.get(
            "chunk_number",
            ""
        )
    )

    print(
        "URL:",
        payload.get(
            "url",
            ""
        )
    )

    print("\nText:")

    print(
        payload.get(
            "text",
            ""
        )
    )


# ============================================================
# SAME-PAGE EXPANSION DISPLAY
# ============================================================

print("\n")
print("=" * 90)
print("SAME-PAGE RELATED CONTENT")
print("=" * 90)


shown_related = set()


for item in expanded_results:

    payload = item["payload"]

    page_url = payload.get(
        "url",
        ""
    )

    anchor_chunk_id = payload.get(
        "chunk_id",
        ""
    )


    related = get_same_page_chunks(
        page_url,
        anchor_chunk_id
    )


    neighbors = find_neighbors(
        payload,
        related
    )


    for neighbor in neighbors:

        neighbor_id = neighbor.get(
            "chunk_id",
            ""
        )


        if neighbor_id in shown_related:
            continue


        shown_related.add(
            neighbor_id
        )


        print("\n" + "-" * 90)

        print(
            "Related page chunk"
        )

        print(
            "Title:",
            neighbor.get(
                "title",
                ""
            )
        )

        print(
            "Section:",
            neighbor.get(
                "section",
                ""
            )
        )

        print(
            "Chunk:",
            neighbor.get(
                "chunk_number",
                ""
            )
        )

        print(
            "URL:",
            neighbor.get(
                "url",
                ""
            )
        )

        print("\nText:")

        print(
            neighbor.get(
                "text",
                ""
            )
        )