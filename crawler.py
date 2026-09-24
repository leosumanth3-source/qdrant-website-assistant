import json
import os
import re
import time
import xml.etree.ElementTree as ET

from collections import deque
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    urldefrag,
)

import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://qdrant.tech/"
DOMAIN = "qdrant.tech"

OUTPUT_DIR = "data/raw"

REQUEST_TIMEOUT = 20
REQUEST_DELAY = 0.5

USER_AGENT = "QdrantWebsiteKnowledgeCrawler/1.0"

# None = continue until there are no more discoverable pages
MAX_PAGES = None


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()

session.headers.update({
    "User-Agent": USER_AGENT
})


# ============================================================
# ROBOTS.TXT
# ============================================================

robots = RobotFileParser()

robots_url = urljoin(
    BASE_URL,
    "/robots.txt"
)

try:
    robots.set_url(robots_url)
    robots.read()

    print("robots.txt loaded.")

except Exception as error:

    print("Could not load robots.txt:")
    print(error)

    print("Continuing without robots.txt rules.")


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):
    """
    Normalize URLs so the same page is not crawled multiple times.

    Removes:
        - URL fragments
        - common tracking parameters
        - duplicate trailing slash
    """

    url = urldefrag(url)[0]

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()

    hostname = parsed.hostname

    if not hostname:
        return ""

    hostname = hostname.lower()

    # Treat www.qdrant.tech as the same site
    if hostname == "www.qdrant.tech":
        hostname = DOMAIN

    path = parsed.path or "/"

    # Remove duplicate trailing slash
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Remove tracking parameters
    if parsed.query:

        query_parts = []

        for part in parsed.query.split("&"):

            key = part.split("=", 1)[0].lower()

            ignored_parameters = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "gclid",
                "fbclid",
                "ref",
            }

            if key not in ignored_parameters:
                query_parts.append(part)

        query = "&".join(query_parts)

    else:

        query = ""

    return urlunparse(
        (
            scheme,
            hostname,
            path,
            "",
            query,
            "",
        )
    )


# ============================================================
# DOMAIN CHECK
# ============================================================

def is_qdrant_url(url):

    parsed = urlparse(url)

    if parsed.scheme not in {
        "http",
        "https",
    }:
        return False

    hostname = parsed.hostname

    if not hostname:
        return False

    hostname = hostname.lower()

    return hostname in {
        "qdrant.tech",
        "www.qdrant.tech",
    }


# ============================================================
# HTML PAGE CHECK
# ============================================================

def is_html_url(url):

    parsed = urlparse(url)

    path = parsed.path.lower()

    blocked_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".webp",
        ".ico",

        ".css",
        ".js",

        ".json",
        ".xml",
        ".txt",

        ".pdf",

        ".zip",
        ".gz",
        ".tar",

        ".mp3",
        ".wav",
        ".mp4",
        ".webm",
        ".mov",

        ".woff",
        ".woff2",
        ".ttf",
        ".otf",

        ".csv",
        ".xlsx",
        ".xls",
        ".doc",
        ".docx",
    )

    return not path.endswith(blocked_extensions)


# ============================================================
# SPECIAL URL CHECK
# ============================================================

def is_ignored_page(url):

    path = urlparse(url).path.lower()

    ignored_paths = (
        "/login",
        "/logout",
        "/signin",
        "/signup",
        "/register",
        "/404",
        "/500",
    )

    for ignored in ignored_paths:

        if path == ignored:
            return True

    return False


# ============================================================
# FINAL URL CHECK
# ============================================================

def is_allowed_url(url):

    if not is_qdrant_url(url):
        return False

    if not is_html_url(url):
        return False

    if is_ignored_page(url):
        return False

    return True


# ============================================================
# PAGE TYPE
# ============================================================

def get_page_type(url):

    path = urlparse(url).path.lower()

    if path == "/":
        return "homepage"

    if "/documentation/" in path:
        return "documentation"

    if path.startswith("/pricing"):
        return "pricing"

    if "/cloud" in path:
        return "cloud"

    if "/blog/" in path:
        return "blog"

    if "/articles/" in path:
        return "article"

    if "/tutorial" in path:
        return "tutorial"

    if "/careers" in path:
        return "careers"

    if "/legal/" in path:
        return "legal"

    if "/about" in path:
        return "about"

    if "/demos" in path:
        return "demo"

    if "/partners" in path:
        return "partners"

    if "/community" in path:
        return "community"

    return "website"


# ============================================================
# SITEMAP URL EXTRACTION
# ============================================================

def parse_sitemap(sitemap_url):

    discovered_urls = set()

    try:

        response = session.get(
            sitemap_url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        print(
            "Failed to download sitemap:",
            sitemap_url
        )

        print(error)

        return discovered_urls


    try:

        root = ET.fromstring(
            response.content
        )

    except ET.ParseError as error:

        print(
            "Failed to parse sitemap:",
            sitemap_url
        )

        print(error)

        return discovered_urls


    namespace = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
    }


    root_tag = root.tag.lower()


    # --------------------------------------------------------
    # Sitemap index
    # --------------------------------------------------------

    if root_tag.endswith("sitemapindex"):

        sitemap_locations = root.findall(
            "sm:sitemap/sm:loc",
            namespace
        )

        print(
            f"Found {len(sitemap_locations)} child sitemaps."
        )


        for loc in sitemap_locations:

            if loc.text:

                child_url = loc.text.strip()

                child_urls = parse_sitemap(
                    child_url
                )

                discovered_urls.update(
                    child_urls
                )


    # --------------------------------------------------------
    # Normal sitemap
    # --------------------------------------------------------

    else:

        locations = root.findall(
            "sm:url/sm:loc",
            namespace
        )

        for loc in locations:

            if not loc.text:
                continue

            url = normalize_url(
                loc.text.strip()
            )

            if is_allowed_url(url):

                discovered_urls.add(url)


    return discovered_urls


# ============================================================
# EXTRACT LINKS FROM HTML
# ============================================================

def extract_internal_links(
    current_url,
    soup
):

    links = set()

    for anchor in soup.find_all(
        "a",
        href=True
    ):

        href = anchor.get("href")

        if not href:
            continue

        href = href.strip()


        # Skip page fragments
        if href.startswith("#"):
            continue

        # Skip email / phone / JS links
        if href.startswith("mailto:"):
            continue

        if href.startswith("tel:"):
            continue

        if href.startswith("javascript:"):
            continue


        full_url = urljoin(
            current_url,
            href
        )


        full_url = normalize_url(
            full_url
        )


        if is_allowed_url(full_url):

            links.add(full_url)


    return links


# ============================================================
# CANONICAL URL
# ============================================================

def get_canonical_url(
    current_url,
    soup
):

    canonical = soup.find(
        "link",
        rel=lambda value:
        value and "canonical" in value
    )

    if canonical and canonical.get("href"):

        canonical_url = urljoin(
            current_url,
            canonical["href"]
        )

        canonical_url = normalize_url(
            canonical_url
        )

        if is_qdrant_url(canonical_url):

            return canonical_url


    return current_url


# ============================================================
# SAVE PAGE
# ============================================================

def save_page(
    page_number,
    url,
    title,
    canonical_url,
    page_type,
    html,
    internal_links,
):

    data = {

        "url": url,

        "canonical_url": canonical_url,

        "title": title,

        "page_type": page_type,

        "internal_links": sorted(
            internal_links
        ),

        "html": html,
    }


    filename = (
        f"page_{page_number:05d}.json"
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


    return output_path


# ============================================================
# DISCOVER INITIAL URLS FROM SITEMAP
# ============================================================

print("\n" + "=" * 70)
print("DISCOVERING URLS FROM SITEMAP")
print("=" * 70)

sitemap_urls = parse_sitemap(
    urljoin(
        BASE_URL,
        "/sitemap.xml"
    )
)

print(
    "Sitemap URLs discovered:",
    len(sitemap_urls)
)


# ============================================================
# INITIAL QUEUE
# ============================================================

queue = deque()

visited = set()


# Put sitemap URLs into queue first
for url in sorted(sitemap_urls):

    queue.append(url)


# Homepage should always be crawled
homepage = normalize_url(
    BASE_URL
)

if homepage not in queue:

    queue.appendleft(
        homepage
    )


print(
    "Initial queue size:",
    len(queue)
)


# ============================================================
# MAIN CRAWLER
# ============================================================

page_number = 0

failed_pages = 0

discovered_via_links = 0


while queue:

    # --------------------------------------------------------
    # MAX PAGE LIMIT
    # --------------------------------------------------------

    if (
        MAX_PAGES is not None
        and len(visited) >= MAX_PAGES
    ):

        print(
            "\nReached MAX_PAGES."
        )

        break


    current_url = queue.popleft()


    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    if current_url in visited:
        continue


    # --------------------------------------------------------
    # URL check
    # --------------------------------------------------------

    if not is_allowed_url(current_url):
        continue


    # --------------------------------------------------------
    # Robots check
    # --------------------------------------------------------

    try:

        if not robots.can_fetch(
            USER_AGENT,
            current_url
        ):

            print(
                "Blocked by robots.txt:",
                current_url
            )

            visited.add(
                current_url
            )

            continue

    except Exception:
        pass


    # --------------------------------------------------------
    # Mark visited
    # --------------------------------------------------------

    visited.add(
        current_url
    )


    print("\n" + "=" * 70)

    print(
        f"Crawling page {len(visited)}"
    )

    print(
        "URL:",
        current_url
    )

    print(
        "Queue:",
        len(queue)
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    try:

        response = session.get(
            current_url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        failed_pages += 1

        print(
            "Request failed:",
            error
        )

        continue


    # --------------------------------------------------------
    # Final URL after redirect
    # --------------------------------------------------------

    final_url = normalize_url(
        response.url
    )


    # --------------------------------------------------------
    # Content type
    # --------------------------------------------------------

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()


    if "text/html" not in content_type:

        print(
            "Skipping non-HTML:",
            content_type
        )

        continue


    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    html = response.text


    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if soup.title:

        title = soup.title.get_text(
            " ",
            strip=True
        )

    else:

        title = ""


    # --------------------------------------------------------
    # Canonical
    # --------------------------------------------------------

    canonical_url = get_canonical_url(
        final_url,
        soup
    )


    # --------------------------------------------------------
    # Page type
    # --------------------------------------------------------

    page_type = get_page_type(
        canonical_url
    )


    # --------------------------------------------------------
    # Internal links
    # --------------------------------------------------------

    internal_links = extract_internal_links(
        final_url,
        soup
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    page_number += 1


    output_path = save_page(
        page_number=page_number,
        url=final_url,
        title=title,
        canonical_url=canonical_url,
        page_type=page_type,
        html=html,
        internal_links=internal_links,
    )


    print(
        "Saved:",
        output_path
    )

    print(
        "Title:",
        title
    )

    print(
        "Type:",
        page_type
    )

    print(
        "Internal links:",
        len(internal_links)
    )


    # --------------------------------------------------------
    # Discover more pages
    # --------------------------------------------------------

    for link in sorted(
        internal_links
    ):

        if link not in visited:

            if link not in queue:

                queue.append(link)

                discovered_via_links += 1


    # --------------------------------------------------------
    # Delay
    # --------------------------------------------------------

    time.sleep(
        REQUEST_DELAY
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest = {

    "base_url": BASE_URL,

    "domain": DOMAIN,

    "pages_saved": page_number,

    "pages_visited": len(visited),

    "failed_pages": failed_pages,

    "discovered_via_links": discovered_via_links,

}


manifest_path = os.path.join(
    OUTPUT_DIR,
    "manifest.json"
)


with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("CRAWLING FINISHED")
print("=" * 70)

print(
    "Pages saved:",
    page_number
)

print(
    "URLs visited:",
    len(visited)
)

print(
    "Failed pages:",
    failed_pages
)

print(
    "Remaining queue:",
    len(queue)
)

print(
    "Manifest:",
    manifest_path
)