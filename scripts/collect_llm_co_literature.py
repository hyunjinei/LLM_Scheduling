#!/usr/bin/env python3
"""Collect metadata, abstract, and method snippets for awesome-fm4co LLM-for-CO papers.

Outputs:
- literature_review/llm_for_co_catalog.json
- literature_review/llm_for_co_catalog.csv

This script is designed for reproducible literature review support. It does not
claim perfect extraction quality; instead, it records what could be collected
from the source URL and whether method text extraction succeeded.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests
from PyPDF2 import PdfReader


AWESOME_FM4CO_RAW = "https://raw.githubusercontent.com/ai4co/awesome-fm4co/main/README.md"
OUT_DIR = Path("literature_review")
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
SOURCE_OVERRIDES = {
    "STARJOB: Dataset for LLM-Driven Job Shop Scheduling": {
        "abstract_url": "https://arxiv.org/abs/2503.01877",
        "method_pdf_url": "https://arxiv.org/pdf/2503.01877",
        "note": "used arXiv mirror instead of OpenReview",
    },
    "ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention": {
        "abstract_url": "https://arxiv.org/abs/2506.11052",
        "method_pdf_url": "https://arxiv.org/pdf/2506.11052",
        "note": "used arXiv mirror instead of OpenReview",
    },
}


@dataclass
class PaperRecord:
    date: str
    title: str
    paper_url: str
    problem: str
    venue: str
    remark: str
    code_url: str
    source_type: str
    abstract_status: str
    abstract: str
    method_status: str
    method_snippet: str
    notes: str


def parse_markdown_url(cell: str) -> tuple[str, str]:
    match = re.match(r"^\[(.*)\]\((.*)\)$", cell)
    if match:
        return match.group(1), match.group(2)
    return cell, cell


def load_llm_for_co_rows() -> list[dict[str, str]]:
    text = requests.get(AWESOME_FM4CO_RAW, timeout=30, headers=USER_AGENT).text
    match = re.search(r"### LLMs for Combinatorial Optimization(.*?)(?:\n### |\Z)", text, re.S)
    if not match:
        raise RuntimeError("Failed to locate the LLMs for Combinatorial Optimization section.")
    section = match.group(1)

    rows = []
    for line in section.splitlines():
        if not line.startswith("| 20"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 6:
            continue
        date, paper_cell, code_cell, problem, venue, remark = parts[:6]
        title, paper_url = parse_markdown_url(paper_cell)
        _, code_url = parse_markdown_url(code_cell) if code_cell else ("", "")
        rows.append(
            {
                "date": date,
                "title": title,
                "paper_url": paper_url,
                "problem": problem.strip("`"),
                "venue": venue.strip("*"),
                "remark": remark,
                "code_url": code_url if code_url != code_cell else "",
            }
        )
    return rows


def detect_source_type(url: str) -> str:
    lowered = url.lower()
    if "arxiv.org" in lowered:
        return "arxiv"
    if "openreview.net" in lowered:
        return "openreview"
    if "doi.org" in lowered or "nature.com" in lowered:
        return "doi_or_publisher"
    return "web"


def arxiv_abs_url(url: str) -> Optional[str]:
    match = re.search(r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    if not match:
        return None
    return f"https://arxiv.org/abs/{match.group(1)}"


def openreview_forum_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    paper_id = query.get("id", [None])[0]
    if not paper_id:
        return None
    return f"https://openreview.net/forum?id={paper_id}"


def extract_meta(html: str, patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, html, re.S | re.I)
        if match:
            return unescape(match.group(1)).strip()
    return None


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=30, headers=USER_AGENT)
    response.raise_for_status()
    return response.text


def fetch_abstract(
    url: str,
    source_type: str,
    override_abstract_url: Optional[str] = None,
) -> tuple[str, str, str]:
    try:
        if override_abstract_url:
            html = fetch_html(override_abstract_url)
            abstract = extract_meta(
                html,
                [
                    r'<meta name="citation_abstract" content="([^"]+)"',
                    r'<meta property="og:description" content="([^"]+)"',
                    r'<meta name="description" content="([^"]+)"',
                ],
            )
            if not abstract:
                return "failed", "", f"override abstract URL has no abstract: {override_abstract_url}"
            return "ok", abstract, ""

        if source_type == "arxiv":
            abs_url = arxiv_abs_url(url)
            if not abs_url:
                return "failed", "", "could not derive arxiv abs URL"
            html = fetch_html(abs_url)
            abstract = extract_meta(
                html,
                [
                    r'<meta name="citation_abstract" content="([^"]+)"',
                    r'<meta property="og:description" content="([^"]+)"',
                    r'<meta name="description" content="([^"]+)"',
                ],
            )
            if not abstract:
                return "failed", "", "no abstract meta tag found"
            return "ok", abstract, ""

        if source_type == "openreview":
            forum_url = openreview_forum_url(url) or url
            html = fetch_html(forum_url)
            abstract = extract_meta(
                html,
                [
                    r'"abstract"\s*:\s*\{\s*"value"\s*:\s*"(.*?)"',
                    r'<meta property="og:description" content="([^"]+)"',
                    r'<meta name="description" content="([^"]+)"',
                ],
            )
            if not abstract:
                return "failed", "", "no abstract found on openreview page"
            return "ok", abstract, ""

        html = fetch_html(url)
        abstract = extract_meta(
            html,
            [
                r'<meta name="citation_abstract" content="([^"]+)"',
                r'<meta property="og:description" content="([^"]+)"',
                r'<meta name="description" content="([^"]+)"',
                r'<meta name="dc\.description" content="([^"]+)"',
            ],
        )
        if not abstract:
            return "failed", "", "no abstract-like meta tag found"
        return "ok", abstract, ""
    except Exception as exc:  # pragma: no cover - network errors are expected
        return "failed", "", str(exc)


def fetch_pdf_text(url: str, max_pages: int = 8) -> str:
    response = requests.get(url, timeout=60, headers=USER_AGENT)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        raise RuntimeError(f"not a PDF response: {content_type}")
    reader = PdfReader(io.BytesIO(response.content))
    pages = []
    for idx in range(min(max_pages, len(reader.pages))):
        pages.append(reader.pages[idx].extract_text() or "")
    return " ".join(" ".join(pages).split())


def find_method_snippet(text: str) -> str:
    lowered = text.lower()
    keywords = [
        " methodology ",
        " method ",
        " methods ",
        " approach ",
        " framework ",
        " proposed method ",
        " training strategy ",
        " inference ",
    ]
    for keyword in keywords:
        index = lowered.find(keyword)
        if index != -1:
            start = max(0, index - 400)
            end = min(len(text), index + 1600)
            return text[start:end].strip()
    return text[:1200].strip()


def fetch_method_snippet(
    url: str,
    source_type: str,
    override_method_pdf_url: Optional[str] = None,
) -> tuple[str, str, str]:
    pdf_url = override_method_pdf_url or url
    if source_type == "arxiv":
        pdf_url = override_method_pdf_url or (url if "/pdf/" in url else url.replace("/abs/", "/pdf/"))
    elif source_type == "openreview":
        parsed = urlparse(url)
        paper_id = parse_qs(parsed.query).get("id", [None])[0]
        if paper_id and not override_method_pdf_url:
            pdf_url = f"https://openreview.net/pdf?id={paper_id}"
    elif source_type not in {"doi_or_publisher", "web"} and not url.lower().endswith(".pdf"):
        return "failed", "", "unsupported source type for method extraction"

    try:
        text = fetch_pdf_text(pdf_url)
        if not text:
            return "failed", "", "empty PDF text"
        return "ok", find_method_snippet(text), ""
    except Exception as exc:  # pragma: no cover - network errors are expected
        return "failed", "", str(exc)


def collect_one(row: dict[str, str]) -> PaperRecord:
    source_type = detect_source_type(row["paper_url"])
    override = SOURCE_OVERRIDES.get(row["title"], {})
    abstract_status, abstract, abstract_note = fetch_abstract(
        row["paper_url"],
        source_type,
        override_abstract_url=override.get("abstract_url"),
    )
    method_status, method_snippet, method_note = fetch_method_snippet(
        row["paper_url"],
        source_type,
        override_method_pdf_url=override.get("method_pdf_url"),
    )
    notes = "; ".join(note for note in [override.get("note", ""), abstract_note, method_note] if note)
    return PaperRecord(
        date=row["date"],
        title=row["title"],
        paper_url=row["paper_url"],
        problem=row["problem"],
        venue=row["venue"],
        remark=row["remark"],
        code_url=row["code_url"],
        source_type=source_type,
        abstract_status=abstract_status,
        abstract=abstract,
        method_status=method_status,
        method_snippet=method_snippet,
        notes=notes,
    )


def main() -> None:
    rows = load_llm_for_co_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    collected: list[PaperRecord] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(collect_one, row): row for row in rows}
        for idx, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            collected.append(record)
            if idx % 10 == 0 or idx == len(rows):
                print(
                    f"[progress] {idx}/{len(rows)} "
                    f"abstract_ok={sum(r.abstract_status == 'ok' for r in collected)} "
                    f"method_ok={sum(r.method_status == 'ok' for r in collected)}"
                )
            time.sleep(0.01)

    collected.sort(key=lambda item: (item.date, item.title))

    json_path = OUT_DIR / "llm_for_co_catalog.json"
    csv_path = OUT_DIR / "llm_for_co_catalog.csv"

    json_path.write_text(
        json.dumps([asdict(record) for record in collected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(asdict(collected[0]).keys()),
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )
        writer.writeheader()
        for record in collected:
            writer.writerow(asdict(record))

    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
