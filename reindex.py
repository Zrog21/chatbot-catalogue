#!/usr/bin/env python3
"""
reindex.py — Table-aware PDF reindexation for AMDP RAG chatbot.

Uses PyMuPDF find_tables() to extract structured table data from catalogues,
creating per-product chunks that preserve table structure.

Usage:
    python reindex.py                          # Reindex default catalogue
    python reindex.py --pdf other.pdf          # Reindex a specific PDF
    python reindex.py --no-embed               # Only generate index, skip embeddings
    python reindex.py --test-pages 340,341,345 # Test extraction on specific pages

Output:
    index_catalogue.json     — chunks with texte + page + type
    embeddings_catalogue.npy — numpy array of 1536d embeddings
"""

import os
import sys
import json
import re
import time
import argparse
import numpy as np
import fitz  # PyMuPDF

# ── Config ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(BASE_DIR, "catalogue.pdf")
INDEX_OUT = os.path.join(BASE_DIR, "index_catalogue.json")
EMBEDDINGS_OUT = os.path.join(BASE_DIR, "embeddings_catalogue.npy")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
EMBED_BATCH = 100

# Chunk size limits
TEXT_CHUNK_SIZE = 600     # Max chars for text chunks
TEXT_CHUNK_OVERLAP = 150  # Overlap for text chunks


def extract_page_context(page):
    """Extract page-level context: section title, drive type, category."""
    text = page.get_text()[:800]
    context_parts = []

    # Detect drive type
    drives = re.findall(r'(\d/\d+)["\u201d\u02ba]?\s*(?:DRIVE|Drive|drive)', text)
    if drives:
        context_parts.append(f'{drives[0]}" Drive')

    # Detect product category from known keywords
    categories = {
        "SOCKET": "Sockets", "RATCHET": "Ratchets", "WRENCH": "Wrenches",
        "SCREWDRIVER": "Screwdrivers", "PLIER": "Pliers", "BIT": "Bits",
        "EXTENSION": "Extensions", "TORQUE": "Torque Tools",
        "ROLL CAB": "Roll Cabs", "TOOL STORAGE": "Tool Storage",
        "HAMMER": "Hammers", "PUNCH": "Punches", "CHISEL": "Chisels",
        "LOW-PROFILE": "Low-Profile", "FLANK DRIVE": "Flank Drive",
        "DEEP": "Deep Socket", "SHALLOW": "Shallow Socket",
    }
    text_upper = text.upper()
    for kw, cat in categories.items():
        if kw in text_upper:
            context_parts.append(cat)
            break

    # Detect section title (usually first big text line)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:5]:
        if line.isupper() and 5 < len(line) < 80 and not line.startswith('\u2022'):
            # Skip generic headers like "ENGLISH" "METRIC" etc
            skip = ["ENGLISH", "METRIC", "INCHES", "MILLIMETERS", "W/ARROWS",
                     "BRITISH STANDARD", "NATURAL", "CHROME", "INDUSTRIAL"]
            if not any(s == line.strip() for s in skip):
                context_parts.append(line)
                break

    return " | ".join(context_parts) if context_parts else ""


def clean_cell(cell):
    """Clean a table cell value."""
    if cell is None:
        return ""
    s = str(cell).strip()
    # Remove excessive whitespace
    s = re.sub(r'\s+', ' ', s)
    # Remove control characters
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    return s


def extract_tables_from_page(page, page_num):
    """Extract all tables from a page as structured chunks."""
    chunks = []
    tables = page.find_tables()

    if not tables.tables:
        return chunks

    page_context = extract_page_context(page)

    for table in tables.tables:
        rows = table.extract()
        if not rows or len(rows) < 2:
            continue

        # First row is typically the header
        headers = [clean_cell(h) for h in rows[0]]
        # Clean empty headers
        headers = [h if h else f"Col{i}" for i, h in enumerate(headers)]

        # Skip tables that are just noise (all empty or single column)
        non_empty_headers = [h for h in headers if h and h != f"Col{0}"]
        if len(non_empty_headers) < 2:
            continue

        # Check if first row is truly a header (contains text, not just numbers)
        first_row_is_header = any(
            bool(re.search(r'[a-zA-Z]{2,}', h)) for h in headers
        )

        if not first_row_is_header:
            # No clear header — treat all rows as data with generic headers
            header_line = ""
            data_rows = rows
        else:
            header_line = " | ".join(h for h in headers if h)
            data_rows = rows[1:]

        # Create one chunk per data row (= one product)
        for row in data_rows:
            cells = [clean_cell(c) for c in row]
            # Skip completely empty rows
            if not any(cells):
                continue

            # Build structured chunk text
            parts = []
            if page_context:
                parts.append(page_context)

            # Pair headers with values, skip empty pairs
            paired = []
            for i, cell in enumerate(cells):
                if cell:
                    h = headers[i] if i < len(headers) and headers[i] != f"Col{i}" else ""
                    if h and h != cell:  # Avoid "A: A" duplicates
                        paired.append(f"{h}: {cell}")
                    else:
                        paired.append(cell)

            if not paired:
                continue

            # Join with pipes for readability
            row_text = " | ".join(paired)
            parts.append(row_text)

            chunk_text = " | ".join(parts)

            # Skip very short chunks (noise)
            if len(chunk_text) < 15:
                continue

            # Also detect if this row has a stock/reference number for enrichment
            stock_refs = re.findall(r'\b[A-Z]{1,6}\d{2,6}[A-Z0-9]*\b', chunk_text)
            if stock_refs:
                chunk_text += f" | Refs: {', '.join(stock_refs[:3])}"

            chunks.append({
                "texte": chunk_text,
                "page": page_num,
                "type": "table_row"
            })

        # Also create a table-summary chunk (headers + first few rows)
        # This helps for broad queries like "what socket sets are available"
        if len(data_rows) >= 3 and header_line:
            summary_rows = []
            for row in data_rows[:5]:
                cells = [clean_cell(c) for c in row]
                if any(cells):
                    summary_rows.append(" | ".join(c for c in cells if c))

            summary_text = f"{page_context}\n{header_line}\n" + "\n".join(summary_rows)
            if len(data_rows) > 5:
                summary_text += f"\n... ({len(data_rows)} products total)"

            chunks.append({
                "texte": summary_text[:TEXT_CHUNK_SIZE],
                "page": page_num,
                "type": "table_summary"
            })

    return chunks


def extract_text_from_page(page, page_num, table_bboxes=None):
    """Extract non-table text from a page as paragraph chunks."""
    chunks = []

    # Get all text blocks
    blocks = page.get_text("blocks")
    # blocks: (x0, y0, x1, y1, text, block_no, block_type)

    text_parts = []
    for block in blocks:
        if block[6] != 0:  # Skip image blocks
            continue
        text = block[4].strip()
        if not text or len(text) < 10:
            continue

        # Skip text that's inside a table bbox
        bx0, by0, bx1, by1 = block[:4]
        in_table = False
        if table_bboxes:
            for tx0, ty0, tx1, ty1 in table_bboxes:
                if bx0 >= tx0 - 5 and by0 >= ty0 - 5 and bx1 <= tx1 + 5 and by1 <= ty1 + 5:
                    in_table = True
                    break
        if in_table:
            continue

        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        text_parts.append(text)

    if not text_parts:
        return chunks

    # Join into full page text (non-table)
    full_text = "\n".join(text_parts)

    # Chunk by paragraphs with overlap
    if len(full_text) <= TEXT_CHUNK_SIZE:
        if len(full_text) >= 30:
            chunks.append({
                "texte": full_text,
                "page": page_num,
                "type": "text"
            })
    else:
        # Split into overlapping chunks
        words = full_text.split()
        current = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= TEXT_CHUNK_SIZE:
                chunk_text = " ".join(current)
                chunks.append({
                    "texte": chunk_text,
                    "page": page_num,
                    "type": "text"
                })
                # Keep overlap
                overlap_words = max(1, TEXT_CHUNK_OVERLAP // 5)
                current = current[-overlap_words:]
                current_len = sum(len(w) + 1 for w in current)

        if current and len(" ".join(current)) >= 30:
            chunks.append({
                "texte": " ".join(current),
                "page": page_num,
                "type": "text"
            })

    return chunks


def process_pdf(pdf_path, test_pages=None):
    """Process a PDF and return all chunks."""
    print(f"Opening {pdf_path}...")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    all_chunks = []
    stats = {"table_rows": 0, "table_summaries": 0, "text": 0, "pages_with_tables": 0}

    pages_to_process = range(total_pages)
    if test_pages:
        pages_to_process = [p - 1 for p in test_pages if 0 < p <= total_pages]
        print(f"Testing pages: {test_pages}")

    for pn in pages_to_process:
        page = doc[pn]
        page_num = pn + 1

        # Extract tables
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables.tables] if tables.tables else []

        if tables.tables:
            stats["pages_with_tables"] += 1
            table_chunks = extract_tables_from_page(page, page_num)
            for c in table_chunks:
                if c["type"] == "table_row":
                    stats["table_rows"] += 1
                else:
                    stats["table_summaries"] += 1
            all_chunks.extend(table_chunks)

        # Extract non-table text
        text_chunks = extract_text_from_page(page, page_num, table_bboxes)
        stats["text"] += len(text_chunks)
        all_chunks.extend(text_chunks)

        # Progress
        if (pn + 1) % 100 == 0 or pn == pages_to_process[-1] if isinstance(pages_to_process, list) else (pn + 1) % 100 == 0:
            print(f"  Processed {pn + 1}/{total_pages} pages, {len(all_chunks)} chunks so far...")

    doc.close()

    print(f"\nExtraction complete:")
    print(f"  Table row chunks: {stats['table_rows']}")
    print(f"  Table summary chunks: {stats['table_summaries']}")
    print(f"  Text chunks: {stats['text']}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Pages with tables: {stats['pages_with_tables']}")

    return all_chunks


def generate_embeddings(chunks, api_key):
    """Generate embeddings for all chunks using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    texts = [c["texte"] for c in chunks]
    all_embeddings = []
    total = len(texts)

    print(f"\nGenerating {EMBED_DIM}d embeddings for {total} chunks...")
    print(f"  Model: {EMBED_MODEL}")
    print(f"  Estimated cost: ~${total * 0.00002:.2f}")

    start = time.time()
    for i in range(0, total, EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        try:
            r = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
                dimensions=EMBED_DIM
            )
            all_embeddings.extend([d.embedding for d in r.data])
        except Exception as e:
            print(f"  ERROR batch {i}: {e}")
            # Fill with zeros for failed batches
            all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch))

        if (i + EMBED_BATCH) % 500 == 0 or i + EMBED_BATCH >= total:
            elapsed = time.time() - start
            pct = min(100, (i + EMBED_BATCH) / total * 100)
            print(f"  {pct:.0f}% ({i + len(batch)}/{total}) - {elapsed:.1f}s")

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"  Done in {time.time() - start:.1f}s — shape: {embeddings.shape}")
    return embeddings


def save_index(chunks, index_path):
    """Save chunks to JSON index."""
    # Save only texte + page (compatible with existing format)
    index_data = [{"texte": c["texte"], "page": c["page"]} for c in chunks]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"Index saved: {index_path} ({size_mb:.1f} MB, {len(index_data)} chunks)")


def save_embeddings(embeddings, emb_path):
    """Save embeddings as numpy binary."""
    np.save(emb_path, embeddings)
    size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"Embeddings saved: {emb_path} ({size_mb:.1f} MB, shape {embeddings.shape})")


def main():
    parser = argparse.ArgumentParser(description="Table-aware PDF reindexation for AMDP RAG")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to PDF catalogue")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--test-pages", type=str, help="Comma-separated page numbers to test")
    parser.add_argument("--index-out", default=INDEX_OUT, help="Output index path")
    parser.add_argument("--emb-out", default=EMBEDDINGS_OUT, help="Output embeddings path")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"ERROR: PDF not found: {args.pdf}")
        sys.exit(1)

    test_pages = None
    if args.test_pages:
        test_pages = [int(p.strip()) for p in args.test_pages.split(",")]

    # Extract chunks
    chunks = process_pdf(args.pdf, test_pages)

    if not chunks:
        print("No chunks extracted!")
        sys.exit(1)

    # Show samples
    print("\n── Sample chunks ──")
    for c in chunks[:5]:
        print(f"  [{c.get('type','?')}] Page {c['page']}: {c['texte'][:120]}...")
    print("  ...")
    # Show a table row sample if exists
    table_rows = [c for c in chunks if c.get("type") == "table_row"]
    if table_rows:
        print(f"\n── Sample table row chunks ──")
        for c in table_rows[:5]:
            print(f"  Page {c['page']}: {c['texte'][:150]}")

    if test_pages:
        print("\nTest mode — not saving.")
        return

    # Save index
    save_index(chunks, args.index_out)

    # Generate and save embeddings
    if not args.no_embed:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\nWARNING: OPENAI_API_KEY not set. Skipping embeddings.")
            print("Set the key and re-run, or use --no-embed to skip.")
        else:
            embeddings = generate_embeddings(chunks, api_key)
            save_embeddings(embeddings, args.emb_out)
    else:
        print("\nSkipping embeddings (--no-embed)")

    print("\nDone!")


if __name__ == "__main__":
    main()
