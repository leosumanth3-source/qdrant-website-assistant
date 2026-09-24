from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ollama import chat


# -----------------------------------
# Configuration
# -----------------------------------

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "qdrant_docs"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "llama3.2:3b"

TOP_K = 3


# -----------------------------------
# Connect to Qdrant
# -----------------------------------

qdrant = QdrantClient(
    url=QDRANT_URL
)


# -----------------------------------
# Load embedding model
# -----------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# -----------------------------------
# Ask user
# -----------------------------------

query = input(
    "\nAsk a question about Qdrant: "
)

print("\nSearching Qdrant...\n")


# -----------------------------------
# Convert question into vector
# -----------------------------------

query_vector = embedding_model.encode(
    query,
    convert_to_numpy=True
).tolist()


# -----------------------------------
# Search Qdrant
# -----------------------------------

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    with_payload=True,
    limit=TOP_K
).points


# -----------------------------------
# Check result
# -----------------------------------

if not results:

    print(
        "No relevant information found."
    )

    exit()


# -----------------------------------
# Get top result
# -----------------------------------

contexts = []

for result in results:

    context = (
        f"Document: {result.payload.get('title', '')}\n"
        f"Section: {result.payload.get('section', '')}\n\n"
        f"{result.payload['text']}"
    )

    contexts.append(context)


context = "\n\n---\n\n".join(
    contexts
)


# -----------------------------------
# Send retrieved text to Ollama
# -----------------------------------

print("Generating answer...\n")


response = chat(

    model=LLM_MODEL,

    messages=[
        {
            "role": "system",

            "content": (
                "You are a Qdrant documentation assistant. "
                "Answer the user's question using only the "
                "documentation context provided below. "
                "Do not use outside knowledge. "
                "Do not invent information. "
                "If the context does not contain enough "
                "information, say: "
                "'The retrieved documentation does not "
                "contain enough information to answer this.'"
            )
        },

        {
            "role": "user",

            "content": (
                f"Question:\n"
                f"{query}\n\n"
                f"Documentation context:\n"
                f"{context}"
            )
        }
    ]
)


# -----------------------------------
# Display final answer
# -----------------------------------

top_result = results[0]

context = top_result.payload["text"]

print("\nRetrieved context:")
print(context)
print("\n" + "=" * 70)