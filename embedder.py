import json
import os

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/chunks"
OUTPUT_DIR = "data/embeddings"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# PROCESS CHUNK FILES
# ============================================================

total_files = 0
total_chunks = 0


for filename in sorted(os.listdir(INPUT_DIR)):

    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    print("\nProcessing:", filename)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        print("No chunks found.")
        continue


    # --------------------------------------------------------
    # Get embedding text
    # --------------------------------------------------------

    texts = []

    for chunk in chunks:

        embedding_text = chunk.get(
            "embedding_text",
            ""
        ).strip()

        if embedding_text:
            texts.append(
                embedding_text
            )


    if not texts:
        print("No embedding text found.")
        continue


    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )


    # --------------------------------------------------------
    # Attach embeddings
    # --------------------------------------------------------

    records = []

    embedding_index = 0

    for chunk in chunks:

        embedding_text = chunk.get(
            "embedding_text",
            ""
        ).strip()

        if not embedding_text:
            continue


        embedding = embeddings[
            embedding_index
        ]

        embedding_index += 1


        record = {
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "url": chunk["url"],
            "canonical_url": chunk["canonical_url"],
            "page_type": chunk["page_type"],
            "breadcrumbs": chunk["breadcrumbs"],
            "section": chunk["section"],
            "section_level": chunk["section_level"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
            "embedding_text": embedding_text,
            "internal_links": chunk["internal_links"],
            "faq_count": chunk["faq_count"],
            "embedding": embedding.tolist()
        }

        records.append(record)


    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    output_filename = filename.replace(
        ".json",
        "_embeddings.json"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False
        )


    total_files += 1
    total_chunks += len(records)


    print(
        "Chunks embedded:",
        len(records)
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EMBEDDING FINISHED")
print("=" * 70)

print(
    "Files:",
    total_files
)

print(
    "Total chunks:",
    total_chunks
)

print(
    "Model:",
    MODEL_NAME
)

print(
    "Vector size: 384"
)

print(
    "Output:",
    OUTPUT_DIR
)