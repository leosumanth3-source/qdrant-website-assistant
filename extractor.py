import json
import os
import re
from urllib.parse import urljoin, urlparse, urldefrag

from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# FIND MAIN CONTENT
# ============================================================

def find_main_content(soup):
    """
    Choose content using priority, not largest size.
    """

    selectors = [
        ".documentation-article",
        ".article-content",
        ".post-content",
        ".markdown-body",
        ".prose",
        "article",
        "[role='main']",
        "main",
    ]

    for selector in selectors:

        element = soup.select_one(selector)

        if not element:
            continue

        text_length = len(
            element.get_text(
                " ",
                strip=True
            )
        )

        if text_length >= 200:
            return element

    return None


# ============================================================
# REMOVE NOISE
# ============================================================

def remove_noise(container):

    selectors = [

        # Technical / browser elements
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",

        # Navigation
        "nav",
        "header",
        "footer",

        # Forms
        "form",

        # Breadcrumbs
        ".breadcrumbs",
        ".breadcrumb",
        "[aria-label='breadcrumb']",
        "[aria-label='Breadcrumb']",

        # Sidebars
        ".sidebar",
        ".documentation-sidebar",
        ".docs-sidebar",

        # Table of contents
        ".toc",
        ".table-of-contents",
        ".table_of_contents",

        # Site navigation
        ".navbar",
        ".navigation",
        ".site-navigation",

        # Cookie / marketing
        ".cookie",
        ".cookies",
        ".cookie-banner",
        ".newsletter",
        ".subscribe",

        # Social / comments
        ".social-share",
        ".share-buttons",
        ".comments",
        ".comment-section",

        # Popups
        ".popup",
        ".modal",

        # Ads
        ".advertisement",
        ".ads",
    ]

    for selector in selectors:

        for element in container.select(selector):
            element.decompose()


# ============================================================
# REMOVE CODE
# ============================================================

def remove_code(container):

    # Actual code blocks
    for element in container.select(
        "pre, code"
    ):
        element.decompose()

    # Common code containers
    for element in container.find_all():

        classes = " ".join(
            element.get("class", [])
        ).lower()

        if any(
            keyword in classes
            for keyword in [
                "code-block",
                "highlight",
                "syntax-highlight",
                "language-",
            ]
        ):
            element.decompose()


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_table(table):

    rows = []

    for tr in table.find_all("tr"):

        cells = tr.find_all(
            ["th", "td"]
        )

        values = []

        for cell in cells:

            value = clean_text(
                cell.get_text(
                    " ",
                    strip=True
                )
            )

            if value:
                values.append(value)

        if values:
            rows.append(values)


    if not rows:
        return ""


    # --------------------------------------------------------
    # Convert table into readable structured text
    # --------------------------------------------------------

    lines = []

    headers = rows[0]

    if len(rows) == 1:

        lines.append(
            " | ".join(headers)
        )

        return "\n".join(lines)


    for row in rows[1:]:

        if len(row) < len(headers):

            row = row + [""] * (
                len(headers) - len(row)
            )


        pairs = []

        for header, value in zip(
            headers,
            row
        ):

            if value:

                pairs.append(
                    f"{header}: {value}"
                )


        if pairs:

            lines.append(
                "; ".join(pairs)
            )


    return "\n".join(lines)


# ============================================================
# EXTRACT BLOCKS
# ============================================================

def extract_blocks(container):

    blocks = []

    elements = container.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "ul",
            "ol",
            "blockquote",
            "table",
            "details",
        ]
    )


    for element in elements:

        # ----------------------------------------------------
        # Skip code
        # ----------------------------------------------------

        if element.find_parent(
            ["pre", "code"]
        ):
            continue


        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if element.name.startswith("h"):

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if not text:
                continue

            level = int(
                element.name[1]
            )

            blocks.append({
                "type": "heading",
                "level": level,
                "text": text
            })

            continue


        # ----------------------------------------------------
        # Paragraph
        # ----------------------------------------------------

        if element.name == "p":

            # Avoid duplicate paragraphs inside lists/details
            if element.find_parent(
                ["li", "details"]
            ):
                continue

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if text:

                blocks.append({
                    "type": "paragraph",
                    "text": text
                })

            continue


        # ----------------------------------------------------
        # Lists
        # ----------------------------------------------------

        if element.name in ["ul", "ol"]:

            # Ignore nested lists
            if element.find_parent(
                ["li", "ul", "ol"]
            ):
                continue

            items = []

            for li in element.find_all(
                "li",
                recursive=False
            ):

                text = clean_text(
                    li.get_text(
                        " ",
                        strip=True
                    )
                )

                if text:
                    items.append(text)

            if items:

                blocks.append({
                    "type": "list",
                    "items": items
                })

            continue


        # ----------------------------------------------------
        # Blockquote
        # ----------------------------------------------------

        if element.name == "blockquote":

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if text:

                blocks.append({
                    "type": "blockquote",
                    "text": text
                })

            continue


        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        if element.name == "table":

            table_text = extract_table(
                element
            )

            if table_text:

                blocks.append({
                    "type": "table",
                    "text": table_text
                })

            continue


        # ----------------------------------------------------
        # FAQ
        # ----------------------------------------------------

        if element.name == "details":

            summary = element.find(
                "summary"
            )

            if not summary:
                continue

            question = clean_text(
                summary.get_text(
                    " ",
                    strip=True
                )
            )

            answer_parts = []

            for child in element.find_all(
                ["p", "li"]
            ):

                text = clean_text(
                    child.get_text(
                        " ",
                        strip=True
                    )
                )

                if text:
                    answer_parts.append(text)

            answer = " ".join(
                answer_parts
            )

            if question and answer:

                blocks.append({
                    "type": "faq",
                    "question": question,
                    "answer": answer
                })

    return blocks


# ============================================================
# EXTRACT BREADCRUMBS
# ============================================================

def extract_breadcrumbs(soup):

    selectors = [
        ".breadcrumbs",
        ".breadcrumb",
        "[aria-label='breadcrumb']",
        "[aria-label='Breadcrumb']",
    ]

    breadcrumbs = []

    for selector in selectors:

        container = soup.select_one(
            selector
        )

        if not container:
            continue

        for element in container.find_all(
            ["a", "span", "li"]
        ):

            text = clean_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if (
                text
                and text not in breadcrumbs
            ):

                breadcrumbs.append(text)

        if breadcrumbs:
            break

    return breadcrumbs


# ============================================================
# EXTRACT LINKS
# ============================================================

def extract_links(
    page_url,
    container
):

    internal_links = []
    external_links = []

    seen_internal = set()
    seen_external = set()


    for anchor in container.find_all(
        "a",
        href=True
    ):

        href = anchor.get(
            "href",
            ""
        ).strip()


        if not href:
            continue


        if href.startswith(
            (
                "#",
                "mailto:",
                "tel:",
                "javascript:"
            )
        ):
            continue


        full_url = urljoin(
            page_url,
            href
        )

        full_url = urldefrag(
            full_url
        )[0]


        parsed = urlparse(
            full_url
        )

        hostname = (
            parsed.hostname or ""
        ).lower()


        link_text = clean_text(
            anchor.get_text(
                " ",
                strip=True
            )
        )


        # ----------------------------------------------------
        # Internal Qdrant links
        # ----------------------------------------------------

        if hostname in {
            "qdrant.tech",
            "www.qdrant.tech",
        }:

            # Remove accidental malformed GitHub paths
            if parsed.path.startswith(
                "/github.com/"
            ):
                continue

            if full_url not in seen_internal:

                seen_internal.add(
                    full_url
                )

                internal_links.append({
                    "text": link_text,
                    "url": full_url
                })


        # ----------------------------------------------------
        # External links
        # ----------------------------------------------------

        else:

            if full_url not in seen_external:

                seen_external.add(
                    full_url
                )

                external_links.append({
                    "text": link_text,
                    "url": full_url
                })


    return internal_links, external_links


# ============================================================
# EXTRACT ONE PAGE
# ============================================================

def extract_page(
    input_path,
    output_path
):

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        page = json.load(file)


    html = page.get(
        "html",
        ""
    )

    url = page.get(
        "url",
        ""
    ).strip()

    canonical_url = page.get(
        "canonical_url",
        url
    ).strip()

    title = page.get(
        "title",
        ""
    ).strip()

    page_type = page.get(
        "page_type",
        "website"
    )


    if not html:
        return False


    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    # --------------------------------------------------------
    # Better title
    # --------------------------------------------------------

    if not title and soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )


    if not title:

        title = "Qdrant Website Page"


    # --------------------------------------------------------
    # Breadcrumbs BEFORE removing them
    # --------------------------------------------------------

    breadcrumbs = extract_breadcrumbs(
        soup
    )


    # --------------------------------------------------------
    # Find real article/content
    # --------------------------------------------------------

    container = find_main_content(
        soup
    )


    if container is None:

        return False


    # --------------------------------------------------------
    # Remove noise
    # --------------------------------------------------------

    remove_noise(
        container
    )

    remove_code(
        container
    )


    # --------------------------------------------------------
    # Extract structured blocks
    # --------------------------------------------------------

    blocks = extract_blocks(
        container
    )


    if not blocks:

        return False


    # --------------------------------------------------------
    # Extract links
    # --------------------------------------------------------

    internal_links, external_links = (
        extract_links(
            url,
            container
        )
    )


    # --------------------------------------------------------
    # Save structured page
    # --------------------------------------------------------

    record = {

        "url": url,

        "canonical_url": canonical_url,

        "title": title,

        "page_type": page_type,

        "breadcrumbs": breadcrumbs,

        "internal_links": internal_links,

        "external_links": external_links,

        "blocks": blocks,

    }


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2
        )


    return True


# ============================================================
# PROCESS ALL RAW PAGES
# ============================================================

processed = 0
failed = 0


for filename in sorted(
    os.listdir(INPUT_DIR)
):

    if not filename.endswith(".json"):
        continue

    if filename == "manifest.json":
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
        "Processing:",
        filename
    )


    try:

        success = extract_page(
            input_path,
            output_path
        )


        if success:

            processed += 1

            print(
                "  Saved:",
                output_filename
                if False
                else filename
            )

        else:

            failed += 1

            print(
                "  No usable content"
            )


    except Exception as error:

        failed += 1

        print(
            "  Failed:",
            error
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURED EXTRACTION FINISHED")
print("=" * 70)

print(
    "Processed:",
    processed
)

print(
    "Failed:",
    failed
)

print(
    "Output:",
    OUTPUT_DIR
)