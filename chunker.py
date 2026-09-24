import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/clean"
OUTPUT_DIR = "data/chunks"

MAX_CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONVERT BLOCK TO TEXT
# ============================================================

def block_to_text(block):
    """
    Convert one structured content block into readable text.
    """

    block_type = block.get("type")

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    if block_type == "heading":

        level = block.get("level", 2)
        text = block.get("text", "").strip()

        return f"{'#' * level} {text}"


    # --------------------------------------------------------
    # Paragraph
    # --------------------------------------------------------

    if block_type == "paragraph":

        return block.get("text", "").strip()


    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if block_type == "list":

        items = block.get("items", [])

        return "\n".join(
            f"- {item}"
            for item in items
            if item.strip()
        )


    # --------------------------------------------------------
    # Blockquote
    # --------------------------------------------------------

    if block_type == "blockquote":

        text = block.get("text", "").strip()

        if text:
            return f"> {text}"

        return ""


    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    if block_type == "table":

        return block.get("text", "").strip()


    # --------------------------------------------------------
    # FAQ
    # --------------------------------------------------------

    if block_type == "faq":

        question = block.get(
            "question",
            ""
        ).strip()

        answer = block.get(
            "answer",
            ""
        ).strip()

        if question and answer:

            return (
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

        return ""


    return ""


# ============================================================
# SPLIT LONG TEXT
# ============================================================

def split_long_text(text, max_size):
    """
    Split very large text into smaller pieces without losing
    the original content.
    """

    if len(text) <= max_size:
        return [text]


    paragraphs = text.split("\n\n")

    pieces = []

    current = ""


    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue


        # ----------------------------------------------------
        # If one paragraph itself is too large
        # ----------------------------------------------------

        if len(paragraph) > max_size:

            if current:
                pieces.append(
                    current.strip()
                )
                current = ""


            start = 0

            while start < len(paragraph):

                end = start + max_size

                piece = paragraph[
                    start:end
                ].strip()

                if piece:
                    pieces.append(piece)

                start = end

            continue


        # ----------------------------------------------------
        # Add paragraph to current piece
        # ----------------------------------------------------

        proposed = (
            current
            + ("\n\n" if current else "")
            + paragraph
        )


        if (
            current
            and len(proposed) > max_size
        ):

            pieces.append(
                current.strip()
            )

            current = paragraph

        else:

            current = proposed


    if current.strip():

        pieces.append(
            current.strip()
        )


    return pieces


# ============================================================
# BUILD SECTION CHUNKS
# ============================================================

def build_chunks(blocks):
    """
    Build section-aware chunks.

    Headings establish the current section.
    Content blocks belong to that section.
    """

    chunks = []

    current_heading = ""
    current_heading_level = 0

    current_blocks = []


    def flush_current_section():

        nonlocal current_blocks

        if not current_blocks:
            return


        # Convert blocks to readable text
        block_texts = []

        for block in current_blocks:

            text = block_to_text(block)

            if text:
                block_texts.append(text)


        if not block_texts:
            current_blocks = []
            return


        section_text = "\n\n".join(
            block_texts
        )


        # Split if too large
        pieces = split_long_text(
            section_text,
            MAX_CHUNK_SIZE
        )


        for piece in pieces:

            chunks.append({
                "section": current_heading,
                "section_level": current_heading_level,
                "text": piece
            })


        current_blocks = []


    # --------------------------------------------------------
    # Process blocks
    # --------------------------------------------------------

    for block in blocks:

        block_type = block.get("type")


        # ----------------------------------------------------
        # New heading
        # ----------------------------------------------------

        if block_type == "heading":

            flush_current_section()


            current_heading = block.get(
                "text",
                ""
            ).strip()

            current_heading_level = block.get(
                "level",
                2
            )

            continue


        # ----------------------------------------------------
        # Content
        # ----------------------------------------------------

        current_blocks.append(
            block
        )


    # --------------------------------------------------------
    # Final section
    # --------------------------------------------------------

    flush_current_section()


    return chunks


# ============================================================
# MERGE VERY SMALL CHUNKS
# ============================================================

def merge_small_chunks(chunks):
    """
    Prevent tiny chunks from becoming isolated pieces.

    Example:
        "Features"
        followed by a 50-character sentence

    These should generally stay together.
    """

    if not chunks:
        return []


    merged = []


    for chunk in chunks:

        text = chunk["text"].strip()

        if not text:
            continue


        # ----------------------------------------------------
        # Merge a very small chunk with previous same section
        # ----------------------------------------------------

        if (
            len(text) < MIN_CHUNK_SIZE
            and merged
            and merged[-1]["section"]
            == chunk["section"]
        ):

            proposed = (
                merged[-1]["text"]
                + "\n\n"
                + text
            )


            if len(proposed) <= MAX_CHUNK_SIZE:

                merged[-1]["text"] = proposed

                continue


        merged.append({
            "section": chunk["section"],
            "section_level": chunk[
                "section_level"
            ],
            "text": text
        })


    return merged


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    input_path,
    output_path
):

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        document = json.load(file)


    # --------------------------------------------------------
    # Page metadata
    # --------------------------------------------------------

    title = document.get(
        "title",
        ""
    ).strip()

    url = document.get(
        "url",
        ""
    ).strip()

    canonical_url = document.get(
        "canonical_url",
        ""
    ).strip()

    page_type = document.get(
        "page_type",
        "website"
    )

    breadcrumbs = document.get(
        "breadcrumbs",
        []
    )

    internal_links = document.get(
        "internal_links",
        []

    )

    faqs = document.get(
        "faqs",
        []
    )

    blocks = document.get(
        "blocks",
        []
    )


    if not blocks:
        return 0


    # --------------------------------------------------------
    # Create section-aware chunks
    # --------------------------------------------------------

    chunks = build_chunks(
        blocks
    )


    chunks = merge_small_chunks(
        chunks
    )


    # --------------------------------------------------------
    # Create final records
    # --------------------------------------------------------

    records = []


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        section = chunk[
            "section"
        ]

        section_level = chunk[
            "section_level"
        ]

        text = chunk[
            "text"
        ].strip()


        if not text:
            continue


        # ----------------------------------------------------
        # Breadcrumb context
        # ----------------------------------------------------

        breadcrumb_text = ""

        if breadcrumbs:

            breadcrumb_text = (
                "Breadcrumbs: "
                + " > ".join(
                    breadcrumbs
                )
                + "\n"
            )


        # ----------------------------------------------------
        # Embedding text
        # ----------------------------------------------------

        embedding_text = (
            f"Document: {title}\n"
            f"Page type: {page_type}\n"
            f"{breadcrumb_text}"
        )


        if section:

            embedding_text += (
                f"Section: {section}\n"
            )


        embedding_text += (
            f"\n{text}"
        )


        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {

            "chunk_id": (
                f"{os.path.splitext(os.path.basename(input_path))[0]}"
                f"_chunk_{index:04d}"
            ),

            "title": title,

            "url": url,

            "canonical_url": canonical_url,

            "page_type": page_type,

            "breadcrumbs": breadcrumbs,

            "section": section,

            "section_level": section_level,

            "chunk_number": index,

            "text": text,

            "embedding_text": embedding_text,

            "internal_links": internal_links,

            "faq_count": len(faqs)
        }


        records.append(
            record
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )


    return len(records)


# ============================================================
# PROCESS ALL CLEAN DOCUMENTS
# ============================================================

processed_documents = 0
failed_documents = 0
total_chunks = 0


for filename in sorted(
    os.listdir(INPUT_DIR)
):

    if not filename.endswith(".json"):
        continue


    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    print(
        "\nProcessing:",
        filename
    )


    try:

        chunk_count = process_document(
            input_path,
            output_path
        )


        if chunk_count == 0:

            failed_documents += 1

            print(
                "  No chunks created."
            )

            continue


        processed_documents += 1

        total_chunks += chunk_count


        print(
            "  Chunks:",
            chunk_count
        )

        print(
            "  Saved:",
            output_path
        )


    except Exception as error:

        failed_documents += 1

        print(
            "  Failed:",
            error
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHUNKING FINISHED")
print("=" * 70)

print(
    "Documents processed:",
    processed_documents
)

print(
    "Documents failed:",
    failed_documents
)

print(
    "Total chunks:",
    total_chunks
)

print(
    "Output directory:",
    OUTPUT_DIR
)