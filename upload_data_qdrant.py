import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(override=True)


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL"
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
)

COLLECTION_NAME = "Qdrantdata"

VECTOR_SIZE = 384

BATCH_SIZE = 100

EMBEDDINGS_DIR = Path(
    "data/embeddings"
)


# ============================================================
# VALIDATION
# ============================================================

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is missing from .env"
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is missing from .env"
    )


# ============================================================
# DISPLAY CONFIG
# ============================================================

print()
print("=" * 90)
print("QDRANT CLOUD UPLOAD")
print("=" * 90)

print(
    "Qdrant URL:",
    QDRANT_URL
)

print(
    "Collection:",
    COLLECTION_NAME
)

print(
    "Vector size:",
    VECTOR_SIZE
)

print(
    "Distance:",
    "COSINE"
)

print(
    "Batch size:",
    BATCH_SIZE
)

print("=" * 90)


# ============================================================
# CONNECT TO QDRANT CLOUD
# ============================================================

print("\nConnecting to Qdrant Cloud...")

try:

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

except Exception as error:

    raise RuntimeError(
        f"Failed to create Qdrant client: {error}"
    )


print("Qdrant client created.")


# ============================================================
# TEST CONNECTION
# ============================================================

print("\nTesting Qdrant connection...")

try:

    collections_response = (
        client.get_collections()
    )

except Exception as error:

    raise RuntimeError(
        "Could not connect to Qdrant Cloud.\n"
        f"Error: {error}"
    )


print("Connection successful.")

existing_collections = {
    collection.name
    for collection
    in collections_response.collections
}


print("\nExisting collections:")

if existing_collections:

    for name in sorted(
        existing_collections
    ):
        print(
            f"  - {name}"
        )

else:

    print(
        "  None"
    )


# ============================================================
# REMOVE EXISTING COLLECTION
# ============================================================
#
# We are migrating the local data into the new Cloud
# collection. Recreating it guarantees the schema is correct.
# ============================================================

if COLLECTION_NAME in existing_collections:

    print()
    print(
        f"Collection '{COLLECTION_NAME}' "
        "already exists."
    )

    print(
        "Deleting existing collection..."
    )

    try:

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to delete existing collection.\n"
            f"Error: {error}"
        )

    print(
        "Existing collection deleted."
    )


# ============================================================
# CREATE COLLECTION
# ============================================================

print()
print(
    f"Creating collection '{COLLECTION_NAME}'..."
)

try:

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=VECTOR_SIZE,

            distance=Distance.COSINE
        )
    )

except Exception as error:

    raise RuntimeError(
        "Failed to create Qdrant collection.\n"
        f"Error: {error}"
    )


print(
    "Collection created successfully."
)


# ============================================================
# FIND EMBEDDING FILES
# ============================================================

print()
print(
    f"Looking for embedding files in: "
    f"{EMBEDDINGS_DIR}"
)

if not EMBEDDINGS_DIR.exists():

    raise RuntimeError(
        f"Embedding directory does not exist: "
        f"{EMBEDDINGS_DIR}"
    )


embedding_files = sorted(
    EMBEDDINGS_DIR.glob(
        "*_embeddings.json"
    )
)


if not embedding_files:

    raise RuntimeError(
        "No embedding files were found."
    )


print(
    f"Found {len(embedding_files)} "
    "embedding files."
)


# ============================================================
# LOAD ALL POINTS
# ============================================================

print()
print("Reading embedding files...")
print("=" * 90)


points = []

total_records = 0

skipped_records = 0


for file_index, file_path in enumerate(
    embedding_files,
    start=1
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception as error:

        print(
            f"[SKIP FILE] "
            f"{file_path.name}: "
            f"{error}"
        )

        continue


    # --------------------------------------------------------
    # Support both:
    #
    # { ... }
    #
    # and:
    #
    # [ {...}, {...} ]
    # --------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        records = [
            data
        ]

    elif isinstance(
        data,
        list
    ):

        records = data

    else:

        print(
            f"[SKIP FILE] "
            f"{file_path.name}: "
            "invalid JSON structure"
        )

        continue


    for record in records:

        total_records += 1


        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding = record.get(
            "embedding"
        )


        if not isinstance(
            embedding,
            list
        ):

            print(
                f"[SKIP RECORD] "
                f"{file_path.name}: "
                "embedding missing or invalid"
            )

            skipped_records += 1

            continue


        # ----------------------------------------------------
        # VECTOR SIZE
        # ----------------------------------------------------

        if len(embedding) != VECTOR_SIZE:

            print(
                f"[SKIP RECORD] "
                f"{file_path.name}: "
                f"expected {VECTOR_SIZE} dimensions, "
                f"got {len(embedding)}"
            )

            skipped_records += 1

            continue


        # ----------------------------------------------------
        # CHUNK ID
        # ----------------------------------------------------

        chunk_id = record.get(
            "chunk_id"
        )


        if not chunk_id:

            print(
                f"[SKIP RECORD] "
                f"{file_path.name}: "
                "chunk_id missing"
            )

            skipped_records += 1

            continue


        chunk_id = str(
            chunk_id
        )


        # ----------------------------------------------------
        # CREATE VALID QDRANT POINT ID
        # ----------------------------------------------------
        #
        # Example:
        #
        # page_00001_chunk_0001
        #
        # becomes:
        #
        # deterministic UUID
        #
        # UUID5 means the same chunk_id always produces
        # the same point ID.
        # ----------------------------------------------------

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id
            )
        )


        # ----------------------------------------------------
        # PAYLOAD
        # ----------------------------------------------------

        payload = {

            "chunk_id":
                record.get(
                    "chunk_id",
                    ""
                ),

            "title":
                record.get(
                    "title",
                    ""
                ),

            "url":
                record.get(
                    "url",
                    ""
                ),

            "canonical_url":
                record.get(
                    "canonical_url",
                    ""
                ),

            "page_type":
                record.get(
                    "page_type",
                    ""
                ),

            "breadcrumbs":
                record.get(
                    "breadcrumbs",
                    []
                ),

            "section":
                record.get(
                    "section",
                    ""
                ),

            "section_level":
                record.get(
                    "section_level",
                    0
                ),

            "chunk_number":
                record.get(
                    "chunk_number",
                    0
                ),

            "text":
                record.get(
                    "text",
                    ""
                ),

            "embedding_text":
                record.get(
                    "embedding_text",
                    ""
                ),

            "internal_links":
                record.get(
                    "internal_links",
                    []
                ),

            "faq_count":
                record.get(
                    "faq_count",
                    0
                ),
        }


        # ----------------------------------------------------
        # QDRANT POINT
        # ----------------------------------------------------

        points.append(

            PointStruct(

                id=point_id,

                vector=embedding,

                payload=payload
            )
        )


    # --------------------------------------------------------
    # PROGRESS READING FILES
    # --------------------------------------------------------

    if (
        file_index % 50 == 0
        or file_index == len(
            embedding_files
        )
    ):

        print(
            f"Processed "
            f"{file_index}/"
            f"{len(embedding_files)} "
            "files..."
        )


# ============================================================
# LOAD SUMMARY
# ============================================================

print()
print("=" * 90)
print("EMBEDDING DATA SUMMARY")
print("=" * 90)

print(
    "Embedding files:",
    len(embedding_files)
)

print(
    "Total records found:",
    total_records
)

print(
    "Skipped records:",
    skipped_records
)

print(
    "Valid points:",
    len(points)
)

print("=" * 90)


# ============================================================
# VALIDATE POINT COUNT
# ============================================================

if not points:

    raise RuntimeError(
        "No valid points are available for upload."
    )


# ============================================================
# UPLOAD IN BATCHES
# ============================================================

print()
print("=" * 90)
print("STARTING UPLOAD")
print("=" * 90)


uploaded = 0

total_points = len(
    points
)


for start in range(
    0,
    total_points,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_points
    )


    batch = points[
        start:end
    ]


    print(
        f"Uploading points "
        f"{start + 1} - {end} "
        f"of {total_points}..."
    )


    try:

        client.upsert(

            collection_name=
                COLLECTION_NAME,

            points=batch,

            wait=True
        )

    except Exception as error:

        print()
        print("=" * 90)
        print("UPLOAD FAILED")
        print("=" * 90)

        print(
            f"Failed batch: "
            f"{start + 1} - {end}"
        )

        print(
            "Error:",
            error
        )

        print("=" * 90)

        raise


    uploaded += len(
        batch
    )


    print(
        f"Uploaded: "
        f"{uploaded}/{total_points}"
    )


# ============================================================
# VERIFY COLLECTION
# ============================================================

print()
print("=" * 90)
print("VERIFYING QDRANT COLLECTION")
print("=" * 90)


try:

    collection_info = (
        client.get_collection(
            collection_name=
                COLLECTION_NAME
        )
    )

except Exception as error:

    raise RuntimeError(
        "Upload completed but collection "
        "verification failed.\n"
        f"Error: {error}"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 90)
print("UPLOAD COMPLETE")
print("=" * 90)

print(
    "Collection:",
    COLLECTION_NAME
)

print(
    "Vector size:",
    VECTOR_SIZE
)

print(
    "Distance:",
    "COSINE"
)

print(
    "Embedding files:",
    len(embedding_files)
)

print(
    "Records found:",
    total_records
)

print(
    "Skipped records:",
    skipped_records
)

print(
    "Uploaded points:",
    uploaded
)

print(
    "Points in Qdrant:",
    collection_info.points_count
)

print("=" * 90)


# ============================================================
# FINAL CHECK
# ============================================================

if collection_info.points_count != uploaded:

    print()
    print(
        "WARNING:"
    )

    print(
        "Uploaded point count and "
        "Qdrant point count do not match."
    )

else:

    print()
    print(
        "SUCCESS: "
        "All uploaded points are present "
        "in Qdrant Cloud."
    )