import os
import json
import re
import base64
print("[BOOT] Module loading started...", flush=True)
import gdown
import fitz
import numpy as np
import httpx
import io
import hashlib
import time
import sqlite3
try:
    import openpyxl as _opxl_global
except ImportError:
    _opxl_global = None
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY      = os.environ.get("PERPLEXITY_API_KEY", "")
LOGIN_USER              = os.environ.get("LOGIN_USER", "amdp")
LOGIN_PASS              = os.environ.get("LOGIN_PASS", "amdp2024")
CATALOGUE_NOM_PRINCIPAL = os.environ.get("CATALOGUE_NOM", "Snap-on Tools")
DRIVE_CATALOGUE  = "1TP1X8JQW02ujnV7CGC44tiBAr6ZjXXqM"
# Template fiche produit AMDP — upload your model PDF to Drive and set this ID
DRIVE_FICHE_TEMPLATE = os.environ.get("DRIVE_FICHE_TEMPLATE", "1NjOSTAnSvAumqT4ryOCEpfR0N6x8JkTA")  # Fiches_techniques.pdf
# Support both legacy 256d and new 1536d — auto-detect at startup
DRIVE_INDEX      = os.environ.get("DRIVE_INDEX",      "1V6H-XKAh9RjbiD8aRttW3YhbSYdOswwk")
DRIVE_EMBEDDINGS = os.environ.get("DRIVE_EMBEDDINGS", "1q4KI9oH4EuqsawJCAEIPnYU6HqAzf4rN")
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PDF_LOCAL        = os.path.join(BASE_DIR, "catalogue.pdf")
FICHE_TEMPLATE   = os.path.join(BASE_DIR, "fiche_template.pdf")
INDEX_LOCAL      = os.path.join(BASE_DIR, "index_catalogue.json")
EMBEDDINGS_LOCAL = os.path.join(BASE_DIR, "embeddings_catalogue.npy")
CHROMA_DIR       = os.path.join(BASE_DIR, "chroma_db")
SQLITE_DB        = os.path.join(BASE_DIR, "excel_data.db")

# Embedding config
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536  # Full dimensions for better accuracy
EMBED_BATCH_SIZE = 100   # Batch embedding calls

client = OpenAI(api_key=OPENAI_API_KEY)
pdfs_supplementaires   = {}
excels_supplementaires = {}

MOTS_LARGE = {"combien","liste","tous","toutes","types","type","gamme","gammes",
              "quels","quelles","existe","disponibles","disponible","ensemble","propose"}

# ── Google Drive download ─────────────────────────────────────────────────────
def telecharger_drive(file_id, destination):
    gdown.download("https://drive.google.com/uc?id=" + file_id, destination, quiet=False, fuzzy=True)

# ── ChromaDB Manager ──────────────────────────────────────────────────────────
class CatalogueVectorDB:
    """Manages ChromaDB collections for all catalogues (main + uploaded)."""

    def __init__(self):
        self.client = None
        self.main_collection = None
        self.main_embed_dim = EMBED_DIMENSIONS  # Track actual dimension used
        self.supp_collections = {}  # filename -> collection
        self.bm25_main = None
        self.bm25_corpus_main = []
        self.bm25_supp = {}  # filename -> (bm25, corpus)
        self.ids_main = []  # Original index data for page references
        self._init_chroma()

    def _init_chroma(self):
        if not HAS_CHROMA:
            print("WARN: chromadb not installed, falling back to numpy")
            return
        self.client = chromadb.Client(ChromaSettings(
            anonymized_telemetry=False,
            is_persistent=False  # In-memory for HF Spaces (fast restart)
        ))
        print("ChromaDB initialized (in-memory)")

    def _embed_texts(self, texts, dimensions=EMBED_DIMENSIONS, batch_size=EMBED_BATCH_SIZE):
        """Batch embed texts via OpenAI API."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                r = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                    dimensions=dimensions
                )
                all_embeddings.extend([d.embedding for d in r.data])
            except Exception as e:
                print(f"Embedding error batch {i}: {e}")
                all_embeddings.extend([[0.0]*dimensions]*len(batch))
        return all_embeddings

    def _build_bm25(self, texts):
        """Build BM25 index from tokenized texts."""
        if not HAS_BM25:
            return None, []
        corpus = [self._tokenize(t) for t in texts]
        try:
            bm25 = BM25Okapi(corpus)
            return bm25, corpus
        except Exception:
            return None, corpus

    def _tokenize(self, text):
        """Simple tokenization for BM25 — lowercase, split, remove short words."""
        text = re.sub(r'[^\w\s/\-]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) >= 2]

    def index_main_catalogue(self, ids_data, legacy_embeddings=None):
        """Index the main catalogue. Uses legacy embeddings if available, else re-embeds."""
        self.ids_main = ids_data
        texts = [item["texte"] for item in ids_data]
        doc_ids = [f"main_{i}" for i in range(len(ids_data))]

        # Build BM25
        print("Building BM25 index for main catalogue...")
        self.bm25_main, self.bm25_corpus_main = self._build_bm25(texts)

        if not self.client:
            # Fallback: keep legacy numpy approach
            if legacy_embeddings is not None:
                self._legacy_embeddings = legacy_embeddings
                self._legacy_ids = ids_data
            return

        # ChromaDB collection
        try:
            self.client.delete_collection("main_catalogue")
        except Exception:
            pass
        self.main_collection = self.client.create_collection(
            name="main_catalogue",
            metadata={"hnsw:space": "cosine"}
        )

        # Use legacy 256d embeddings converted, or re-embed
        if legacy_embeddings is not None and len(legacy_embeddings) == len(texts):
            self.main_embed_dim = legacy_embeddings.shape[1]  # Remember actual dimension
            self._legacy_embeddings = legacy_embeddings  # Keep for fallback
            self._legacy_ids = ids_data
            print(f"Using legacy embeddings ({self.main_embed_dim}d) for {len(texts)} chunks...")
            # Add in batches
            for i in range(0, len(texts), 500):
                end = min(i+500, len(texts))
                self.main_collection.add(
                    ids=doc_ids[i:end],
                    documents=texts[i:end],
                    embeddings=[legacy_embeddings[j].tolist() for j in range(i, end)],
                    metadatas=[{"page": ids_data[j]["page"], "idx": j} for j in range(i, end)]
                )
            print(f"Main catalogue indexed: {len(texts)} chunks (legacy {legacy_embeddings.shape[1]}d)")
        else:
            print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ({EMBED_DIMENSIONS}d)...")
            embeddings = self._embed_texts(texts)
            for i in range(0, len(texts), 500):
                end = min(i+500, len(texts))
                self.main_collection.add(
                    ids=doc_ids[i:end],
                    documents=texts[i:end],
                    embeddings=embeddings[i:end],
                    metadatas=[{"page": ids_data[j]["page"], "idx": j} for j in range(i, end)]
                )
            print(f"Main catalogue indexed: {len(texts)} chunks ({EMBED_DIMENSIONS}d)")

    def index_supplementary_pdf(self, filename, chunks):
        """Index an uploaded PDF with full embeddings + BM25."""
        texts = [c["texte"] for c in chunks]
        if not texts:
            return

        # BM25
        bm25, corpus = self._build_bm25(texts)
        self.bm25_supp[filename] = (bm25, corpus, chunks)

        if not self.client:
            return

        col_name = "supp_" + hashlib.md5(filename.encode()).hexdigest()[:12]
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Embedding supplementary PDF '{filename}': {len(texts)} chunks...")
        embeddings = self._embed_texts(texts)
        doc_ids = [f"supp_{i}" for i in range(len(texts))]

        for i in range(0, len(texts), 500):
            end = min(i+500, len(texts))
            collection.add(
                ids=doc_ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=[{"page": chunks[j].get("page", 0), "idx": j} for j in range(i, end)]
            )

        self.supp_collections[filename] = (collection, chunks)
        print(f"Supplementary PDF indexed: {len(texts)} chunks")

    def search_main(self, query_text, query_en, n=20, reference=None, specs=None):
        """Hybrid search: ChromaDB semantic + BM25 textual, with fusion."""
        question_lower = query_text.lower()

        # ── Semantic search ──
        sem_scores = {}
        if self.client and self.main_collection:
            try:
                # Search with both FR and EN queries for better recall
                queries = [query_en]
                if query_text.lower() != query_en.lower():
                    queries.append(query_text)

                query_embeddings = self._embed_texts(queries, dimensions=self.main_embed_dim)
                for qe in query_embeddings:
                    results = self.main_collection.query(
                        query_embeddings=[qe],
                        n_results=min(n * 2, 40)
                    )
                    if results and results["ids"] and results["ids"][0]:
                        for idx_str, dist in zip(results["ids"][0], results["distances"][0]):
                            i = int(idx_str.replace("main_", ""))
                            score = max(0, 1 - dist)  # cosine distance to similarity
                            sem_scores[i] = max(sem_scores.get(i, 0), score)
            except Exception as e:
                print(f"ChromaDB search error: {e}")
                # Fallback to legacy numpy if available
                sem_scores = self._legacy_semantic_search(query_en, n * 2)
        else:
            sem_scores = self._legacy_semantic_search(query_en, n * 2)

        # ── BM25 textual search ──
        bm25_scores = {}
        if self.bm25_main and HAS_BM25:
            # Tokenize query (both FR and EN)
            tokens_fr = self._tokenize(query_text)
            tokens_en = self._tokenize(query_en)
            tokens = list(set(tokens_fr + tokens_en))

            if tokens:
                scores = self.bm25_main.get_scores(tokens)
                max_bm25 = max(scores) if max(scores) > 0 else 1
                for i, s in enumerate(scores):
                    if s > 0:
                        bm25_scores[i] = s / max_bm25
        else:
            # Fallback: old textual search
            bm25_scores = self._legacy_textual_search(query_text, reference, specs)

        # ── Reference & spec boosting ──
        boost_scores = {}
        if reference or specs:
            for idx, item in enumerate(self.ids_main):
                texte = item["texte"].lower()
                boost = 0
                if reference and reference.lower() in texte:
                    boost += 1.0  # Strong boost for exact reference match
                if specs:
                    for sp in specs:
                        if sp in texte:
                            boost += 0.3
                        for variant in [sp+"mm", sp+" mm", sp+"dents", sp+"/4", sp+"/8"]:
                            if variant in texte:
                                boost += 0.2
                if boost > 0:
                    boost_scores[idx] = min(boost, 1.0)

        # ── Fusion ──
        # Adaptive weights based on query type
        if reference:
            w_sem, w_bm25, w_boost = 0.2, 0.3, 0.5
        elif specs:
            w_sem, w_bm25, w_boost = 0.3, 0.4, 0.3
        else:
            w_sem, w_bm25, w_boost = 0.5, 0.5, 0.0

        all_ids = set(sem_scores.keys()) | set(bm25_scores.keys()) | set(boost_scores.keys())
        fused = []
        for idx in all_ids:
            score = (w_sem * sem_scores.get(idx, 0)
                   + w_bm25 * bm25_scores.get(idx, 0)
                   + w_boost * boost_scores.get(idx, 0))
            fused.append((idx, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [self.ids_main[idx] for idx, _ in fused[:n]]

    def search_supplementary(self, filename, query_text, query_en, n=10):
        """Search in a supplementary PDF using hybrid search."""
        results = []

        # ChromaDB semantic
        sem_results = {}
        if filename in self.supp_collections:
            collection, chunks = self.supp_collections[filename]
            try:
                query_emb = self._embed_texts([query_en])[0]
                res = collection.query(query_embeddings=[query_emb], n_results=min(n*2, 20))
                if res and res["ids"] and res["ids"][0]:
                    for idx_str, dist in zip(res["ids"][0], res["distances"][0]):
                        i = int(idx_str.replace("supp_", ""))
                        sem_results[i] = max(0, 1 - dist)
            except Exception as e:
                print(f"Supp search error: {e}")

        # BM25
        bm25_results = {}
        if filename in self.bm25_supp and HAS_BM25:
            bm25, corpus, chunks = self.bm25_supp[filename]
            if bm25:
                tokens = self._tokenize(query_text) + self._tokenize(query_en)
                tokens = list(set(tokens))
                if tokens:
                    scores = bm25.get_scores(tokens)
                    max_s = max(scores) if max(scores) > 0 else 1
                    for i, s in enumerate(scores):
                        if s > 0:
                            bm25_results[i] = s / max_s

        # Get chunks reference
        if filename in self.bm25_supp:
            _, _, chunks = self.bm25_supp[filename]
        elif filename in self.supp_collections:
            _, chunks = self.supp_collections[filename]
        else:
            return []

        # Fusion 50/50
        all_ids = set(sem_results.keys()) | set(bm25_results.keys())
        fused = [(i, 0.5*sem_results.get(i,0) + 0.5*bm25_results.get(i,0)) for i in all_ids]
        fused.sort(key=lambda x: x[1], reverse=True)
        return [chunks[i] for i, _ in fused[:n]]

    def search_all_supplementary(self, query_text, query_en, n=5):
        """Search across all supplementary PDFs."""
        all_results = []
        filenames = set(list(self.supp_collections.keys()) + list(self.bm25_supp.keys()))
        for fn in filenames:
            results = self.search_supplementary(fn, query_text, query_en, n=3)
            for r in results:
                r["_source_file"] = fn
            all_results.extend(results)
        return all_results[:n]

    def _legacy_semantic_search(self, query_en, n=35):
        """Fallback: numpy cosine similarity with legacy 256d embeddings."""
        if not hasattr(self, '_legacy_embeddings') or self._legacy_embeddings is None:
            return {}
        try:
            emb = self._legacy_embeddings
            dim = emb.shape[1] if len(emb.shape) > 1 else 256
            r = client.embeddings.create(model=EMBED_MODEL, input=[query_en], dimensions=dim)
            q = np.array(r.data[0].embedding, dtype=np.float32)
            normes = np.linalg.norm(emb, axis=1)
            scores = np.dot(emb, q) / (normes * np.linalg.norm(q) + 1e-10)
            top = np.argsort(scores)[::-1][:n]
            return {int(i): float(scores[i]) for i in top}
        except Exception as e:
            print(f"Legacy semantic search error: {e}")
            return {}

    def _legacy_textual_search(self, question, reference=None, specs=None):
        """Fallback: old-style word counting."""
        mots = [m.strip(".,;:?!()").lower() for m in question.split() if len(m) >= 2]
        scores = {}
        for idx, item in enumerate(self.ids_main):
            texte = item["texte"].lower()
            score = 0
            for m in mots:
                cnt = texte.count(m)
                if cnt > 0:
                    score += min(cnt, 3)
            if reference and reference.lower() in texte:
                score += 25
            if specs:
                for sp in specs:
                    if sp in texte:
                        score += 8
                    for variant in [sp+"mm", sp+" mm", sp+"dents", sp+"/4", sp+"/8"]:
                        if variant in texte:
                            score += 5
            if score > 0:
                scores[idx] = score
        max_score = max(scores.values()) if scores else 1
        return {i: scores[i] / max_score for i in scores}


# Global vector DB
vector_db = CatalogueVectorDB()

# ── PDF Chunking ──────────────────────────────────────────────────────────────
def extraire_texte_pdf(contenu_bytes):
    """Extract text from PDF, page by page."""
    texte_pages = []
    doc = fitz.open(stream=contenu_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        texte = page.get_text()
        if texte.strip():
            texte_pages.append("[Page " + str(i+1) + "]\n" + texte.strip())
    doc.close()
    return "\n\n".join(texte_pages)

def chunker_pdf_intelligent(contenu_bytes, chunk_size=600, overlap=150):
    """
    Table-aware chunking: extracts structured table data + text from PDF.
    - Tables: one chunk per row (preserves product data structure)
    - Text: paragraph-based chunks with overlap
    """
    doc = fitz.open(stream=contenu_bytes, filetype="pdf")
    chunks = []

    for i, page in enumerate(doc):
        page_num = i + 1

        # ── Extract page context ──
        page_text = page.get_text()[:500]
        context_parts = []
        drives = re.findall(r'(\d/\d+)["\u201d\u02ba]?\s*(?:DRIVE|Drive|drive)', page_text)
        if drives:
            context_parts.append(f'{drives[0]}" Drive')
        lines = [l.strip() for l in page_text.split('\n') if l.strip()]
        for line in lines[:5]:
            if line.isupper() and 5 < len(line) < 80:
                context_parts.append(line)
                break
        page_context = " | ".join(context_parts)

        # ── Extract tables ──
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables.tables] if tables.tables else []

        for table in (tables.tables if tables.tables else []):
            rows = table.extract()
            if not rows or len(rows) < 2:
                continue
            headers = [str(h).strip() if h else "" for h in rows[0]]
            headers = [h if h else f"Col{j}" for j, h in enumerate(headers)]
            first_row_is_header = any(bool(re.search(r'[a-zA-Z]{2,}', h)) for h in headers)
            data_rows = rows[1:] if first_row_is_header else rows

            for row in data_rows:
                cells = [re.sub(r'\s+', ' ', str(c).strip()) if c else "" for c in row]
                if not any(cells):
                    continue
                parts = []
                if page_context:
                    parts.append(page_context)
                paired = []
                for ci, cell in enumerate(cells):
                    if cell:
                        h = headers[ci] if ci < len(headers) and headers[ci] != f"Col{ci}" else ""
                        if h and h != cell:
                            paired.append(f"{h}: {cell}")
                        else:
                            paired.append(cell)
                if not paired:
                    continue
                parts.append(" | ".join(paired))
                chunk_text = " | ".join(parts)
                if len(chunk_text) >= 15:
                    chunks.append({"texte": chunk_text, "page": page_num})

        # ── Extract non-table text ──
        blocks = page.get_text("blocks")
        text_parts = []
        for block in blocks:
            if block[6] != 0:
                continue
            text = block[4].strip()
            if not text or len(text) < 10:
                continue
            bx0, by0, bx1, by1 = block[:4]
            in_table = any(
                bx0 >= tx0 - 5 and by0 >= ty0 - 5 and bx1 <= tx1 + 5 and by1 <= ty1 + 5
                for tx0, ty0, tx1, ty1 in table_bboxes
            )
            if in_table:
                continue
            text_parts.append(re.sub(r'\s+', ' ', text))

        full_text = "\n".join(text_parts)
        if len(full_text) < 30:
            continue

        if len(full_text) <= chunk_size:
            chunks.append({"texte": full_text, "page": page_num})
        else:
            words = full_text.split()
            current = []
            current_len = 0
            for word in words:
                current.append(word)
                current_len += len(word) + 1
                if current_len >= chunk_size:
                    chunks.append({"texte": " ".join(current), "page": page_num})
                    overlap_words = max(1, overlap // 5)
                    current = current[-overlap_words:]
                    current_len = sum(len(w) + 1 for w in current)
            if current and len(" ".join(current)) >= 30:
                chunks.append({"texte": " ".join(current), "page": page_num})

    doc.close()
    return chunks

# ── Helpers ───────────────────────────────────────────────────────────────────
def detecter_reference(question):
    for t in question.split():
        c = t.strip(".,;:?!()")
        if re.match(r'^[A-Za-z]{1,6}[0-9]{1,6}[A-Za-z0-9]*$', c) and len(c) >= 4:
            unites = ['nm','mm','cm','kg','lb','oz','ft','in','rpm','bar','psi','db','hz']
            if c.lower() not in unites and not any(c.lower().endswith(u) and c[:-len(u)].isdigit() for u in unites):
                return c
        if re.match(r'^[0-9]{4,}[A-Za-z0-9]*$', c) and len(c) >= 5:
            return c
    return None

def extraire_specs(question):
    specs = []
    for t in question.split():
        c = t.strip(".,;:?!()")
        if re.match(r'^[0-9]+$', c) and 2 <= len(c) <= 4:
            specs.append(c)
        if re.match(r'^[0-9]+/[0-9]+$', c):
            specs.append(c)
    return specs

def traduire_query(question):
    """Translate query FR->EN for semantic search — generic, not brand-specific."""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=100,
            messages=[
                {"role":"system","content":
                    "You are a professional tool/hardware/industrial equipment translator FR->EN. "
                    "Translate the French query to English for semantic search in a professional tools catalog. "
                    "Handle all brands and types: hand tools, power tools, diagnostics, storage, automotive, industrial. "
                    "Keep product codes and reference numbers unchanged. "
                    "Also output key synonyms if relevant (e.g. 'ratchet wrench' -> also 'ratchet, socket wrench'). "
                    "Reply ONLY with the translation, nothing else."},
                {"role":"user","content":question}
            ])
        return r.choices[0].message.content.strip()
    except Exception:
        return question

def reranker_gpt(question, chunks, top_n=8, query_en=""):
    """
    Use GPT-4o-mini to rerank chunks by relevance to the question.
    Returns the top_n most relevant chunks in order.
    NEVER returns empty — always falls back to original order.
    """
    if not chunks or len(chunks) <= top_n:
        return chunks

    # Prepare chunk summaries for reranking (limit text length)
    chunk_summaries = []
    for i, c in enumerate(chunks[:20]):  # Max 20 to rerank
        text = c["texte"][:300]  # Truncate for token efficiency
        chunk_summaries.append(f"[{i}] Page {c.get('page','?')}: {text}")

    # Include EN translation in the prompt so GPT can match FR question to EN chunks
    question_display = question
    if query_en and query_en.lower() != question.lower():
        question_display = f"{question} (EN: {query_en})"

    prompt = (
        f"Question utilisateur: {question_display}\n\n"
        f"IMPORTANT: Les extraits sont en ANGLAIS et la question peut être en FRANÇAIS. "
        f"Tu dois évaluer la pertinence TECHNIQUE du contenu, pas la correspondance linguistique. "
        f"Un extrait sur '1/4 drive socket 10mm' EST pertinent pour 'douille 1/4 de 10mm'.\n\n"
        f"Voici {len(chunk_summaries)} extraits de catalogue. "
        f"Classe les {top_n} extraits les PLUS PERTINENTS pour repondre a la question. "
        f"Reponds UNIQUEMENT avec les numeros des extraits, separes par des virgules, "
        f"du plus pertinent au moins pertinent.\n\n"
        + "\n\n".join(chunk_summaries)
    )

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=100,
            messages=[
                {"role":"system","content":
                    "Tu es un expert en outillage professionnel et en pertinence documentaire. "
                    "Le catalogue est en anglais, les questions sont souvent en français. "
                    "Tu dois faire le lien TECHNIQUE entre les termes FR et EN "
                    "(douille=socket, cliquet=ratchet, clé=wrench, tournevis=screwdriver, etc). "
                    "Retourne TOUJOURS au moins 3 numeros d'extraits, meme si la pertinence est faible."},
                {"role":"user","content":prompt}
            ]
        )
        response = r.choices[0].message.content.strip()
        print(f"[RERANKER] Response: {response}")
        # Parse indices — ignore "AUCUN" and always try to extract numbers
        indices = []
        for part in re.findall(r'\d+', response):
            idx = int(part)
            if 0 <= idx < len(chunks[:20]) and idx not in indices:
                indices.append(idx)
        if len(indices) >= 2:
            return [chunks[i] for i in indices[:top_n]]
        # If reranker returned too few results, fall through to fallback
        print(f"[RERANKER] Trop peu de résultats ({len(indices)}), fallback ordre original")
    except Exception as e:
        print(f"Reranker error: {e}")

    # Fallback: return first top_n (original hybrid search order)
    return chunks[:top_n]

def page_en_image_base64(numero_page):
    doc   = fitz.open(PDF_LOCAL)
    page  = doc[numero_page - 1]
    pix   = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    image_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(image_bytes).decode("utf-8")

# ── Excel SQLite Database ─────────────────────────────────────────────────────
class ExcelSQLiteDB:
    """Stores Excel data in SQLite for powerful Text-to-SQL queries."""

    def __init__(self, db_path=SQLITE_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.tables = {}  # filename -> table_name

    def _sanitize_col(self, col):
        """Sanitize column name for SQL."""
        col = re.sub(r'[^\w]', '_', str(col).strip())
        col = col.strip('_') or 'col'
        if col[0].isdigit():
            col = 'c_' + col
        return col.lower()

    def _table_name(self, filename):
        """Generate safe table name from filename."""
        name = re.sub(r'[^\w]', '_', filename.rsplit('.', 1)[0])
        return 'excel_' + name.lower()[:50]

    def import_excel(self, filename, lignes_txt, nom_affichage):
        """Import Excel/CSV lines into SQLite table with auto-detected schema."""
        table = self._table_name(filename)

        # Parse lines to find header and data
        lines = [l for l in lignes_txt if l.strip() and not l.startswith('[Feuille:')]
        if not lines:
            print(f"SQLite: no data lines found in {filename}")
            return None

        # Debug: show what we received
        print(f"SQLite: importing {filename} — {len(lines)} lines")
        for i, l in enumerate(lines[:3]):
            print(f"  Line {i}: {l[:100]}")

        # First non-empty line = header
        header_line = lines[0]
        sep = ' | ' if ' | ' in header_line else '\t' if '\t' in header_line else ','
        raw_cols = header_line.split(sep)
        cols = [self._sanitize_col(c) for c in raw_cols]

        # If all cols are empty/generic, generate col names
        if all(c in ('col', 'c_', '') for c in cols):
            cols = [f'col_{i}' for i in range(len(raw_cols))]

        # Deduplicate column names
        seen = {}
        unique_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                unique_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                unique_cols.append(c)
        cols = unique_cols

        # Drop old table if exists
        self.conn.execute(f'DROP TABLE IF EXISTS "{table}"')

        # Create table
        col_defs = ', '.join([f'"{c}" TEXT' for c in cols])
        self.conn.execute(f'CREATE TABLE "{table}" ({col_defs})')

        # Insert data rows
        n_inserted = 0
        for line in lines[1:]:
            values = line.split(sep)
            # Pad or truncate to match column count
            values = values[:len(cols)]
            while len(values) < len(cols):
                values.append('')
            values = [v.strip() for v in values]
            placeholders = ','.join(['?'] * len(cols))
            try:
                self.conn.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', values)
                n_inserted += 1
            except Exception as e:
                print(f"  Insert error: {e} — values: {values[:3]}...")

        self.conn.commit()
        self.tables[filename] = {
            'table': table,
            'columns': cols,
            'raw_columns': [c.strip() for c in raw_cols],
            'rows': n_inserted,
            'nom_affichage': nom_affichage
        }
        print(f"SQLite: imported '{nom_affichage}' -> {table} ({n_inserted} rows, {len(cols)} cols: {cols})")

        # Debug: verify data
        try:
            verify = self.conn.execute(f'SELECT * FROM "{table}" LIMIT 2').fetchall()
            for row in verify:
                print(f"  Verify: {' | '.join(str(v) for v in row)}")
        except Exception:
            pass

        return table

    def query_with_llm(self, question, reference=None):
        """Use GPT to generate SQL query, execute it, return results."""
        if not self.tables:
            return None, []

        # Build schema description for LLM
        schema_desc = ""
        for fn, info in self.tables.items():
            schema_desc += f"Table: {info['table']} (fichier: {info['nom_affichage']})\n"
            schema_desc += f"  Colonnes: {', '.join(info['columns'])}\n"
            schema_desc += f"  Colonnes originales: {', '.join(info['raw_columns'])}\n"
            schema_desc += f"  ({info['rows']} lignes)\n"
            # Show ALL rows if small table, else first 5
            max_example = info['rows'] if info['rows'] <= 30 else 5
            try:
                rows = self.conn.execute(f'SELECT * FROM "{info["table"]}" LIMIT {max_example}').fetchall()
                for row in rows:
                    schema_desc += f"  Ligne: {' | '.join(str(v) for v in row)}\n"
            except Exception:
                pass
            schema_desc += "\n"

        # Ask LLM to generate SQL
        try:
            r = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=300,
                messages=[
                    {"role": "system", "content":
                        "Tu es un expert SQL. Genere une requete SQLite pour repondre a la question. "
                        "Reponds UNIQUEMENT avec la requete SQL, rien d'autre. Pas de markdown, pas de ```. "
                        "Utilise LIKE avec % pour les recherches textuelles. "
                        "Utilise LOWER() pour rendre les recherches insensibles a la casse. "
                        "IMPORTANT: les references produit peuvent contenir des espaces dans la base (ex: 'TL72 FOD' au lieu de 'TL72FOD'). "
                        "Pour chercher une reference, utilise REPLACE pour supprimer les espaces: LOWER(REPLACE(colonne,' ','')) LIKE LOWER(REPLACE('%terme%',' ','')). "
                        "Si la question mentionne une reference produit, cherche dans TOUTES les colonnes avec OR. "
                        "Limite les resultats a 20 lignes max. "
                        "IMPORTANT: utilise les noms de colonnes EXACTS du schema."},
                    {"role": "user", "content":
                        f"Schema des tables:\n{schema_desc}\n"
                        f"Question: {question}\n"
                        f"{'Reference detectee: ' + reference if reference else ''}\n"
                        f"Requete SQL:"}
                ]
            )
            sql = r.choices[0].message.content.strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            # Remove any non-SQL text
            if sql.upper().startswith('SELECT') or sql.upper().startswith('WITH'):
                pass
            else:
                # Try to find SELECT in the response
                match = re.search(r'(SELECT\s.+)', sql, re.IGNORECASE | re.DOTALL)
                if match:
                    sql = match.group(1)
            print(f"[SQL] Generated: {sql}")

            # Execute with error handling
            cursor = self.conn.execute(sql)
            col_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            if rows:
                results_text = " | ".join(col_names) + "\n"
                results_text += "-" * 40 + "\n"
                for row in rows:
                    results_text += " | ".join(str(v) for v in row) + "\n"
                return results_text, list(self.tables.values())
            else:
                # SQL worked but no results — try fallback
                print("[SQL] Query returned 0 rows, trying fallback...")
                return self._fallback_search(reference or question)

        except Exception as e:
            print(f"[SQL] Error: {e}")
            # Fallback: brute force search
            return self._fallback_search(reference or question)

    def _fallback_search(self, search_term):
        """Brute-force search for a term across all tables and all columns."""
        if not search_term:
            return None, []
        term_lower = search_term.lower().strip()
        # Build multiple search variants
        words = [w.strip(".,;:?!()").lower() for w in search_term.split() if len(w) >= 2]
        # Remove spaces from term for matching "TL72 FOD" with "TL72FOD"
        term_nospace = term_lower.replace(" ", "").replace("-", "")
        search_terms = [term_lower, term_nospace] + words

        # Add sub-parts for fuzzy matching (e.g. "T72FOD" -> "72FOD", "T72", "FOD")
        # Extract numeric part and alpha parts
        nums = re.findall(r'\d+', term_nospace)
        alphas = re.findall(r'[a-zA-Z]+', term_nospace)
        for n in nums:
            if len(n) >= 2:
                search_terms.append(n)
                # Combine with surrounding letters
                for a in alphas:
                    if len(a) >= 2:
                        search_terms.append(n + a)
                        search_terms.append(a + n)

        # Deduplicate
        seen = set()
        unique_terms = []
        for t in search_terms:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                unique_terms.append(t)
        search_terms = unique_terms

        print(f"[SQL fallback] Searching with terms: {search_terms}")

        for fn, info in self.tables.items():
            for term in search_terms:
                try:
                    # Search with REPLACE to also strip spaces in DB values
                    conditions = ' OR '.join([
                        f'LOWER("{c}") LIKE ? OR LOWER(REPLACE("{c}"," ","")) LIKE ?'
                        for c in info['columns']
                    ])
                    params = []
                    for c in info['columns']:
                        params.extend([f'%{term}%', f'%{term}%'])
                    sql = f'SELECT * FROM "{info["table"]}" WHERE {conditions} LIMIT 10'
                    rows = self.conn.execute(sql, params).fetchall()
                    if rows:
                        col_names = info['columns']
                        results_text = " | ".join(col_names) + "\n"
                        results_text += "-" * 40 + "\n"
                        for row in rows:
                            results_text += " | ".join(str(v) for v in row) + "\n"
                        print(f"[SQL fallback] Found {len(rows)} rows with term '{term}'")
                        return results_text, [info]
                except Exception as e:
                    print(f"[SQL fallback] Error: {e}")
        return None, []

    def find_similar_refs(self, reference, max_results=5):
        """Find similar references in all tables (fuzzy matching)."""
        if not reference or not self.tables:
            return []
        ref_clean = reference.lower().replace(" ", "").replace("-", "")
        similar = []

        for fn, info in self.tables.items():
            try:
                # Get all values from all columns
                for col in info['columns']:
                    rows = self.conn.execute(f'SELECT DISTINCT "{col}" FROM "{info["table"]}"').fetchall()
                    for row in rows:
                        val = str(row[0]).strip()
                        if not val or len(val) < 2:
                            continue
                        val_clean = val.lower().replace(" ", "").replace("-", "")
                        # Check if reference is a substring or similar
                        if ref_clean in val_clean or val_clean in ref_clean:
                            if val not in similar:
                                similar.append(val)
                        # Check common prefix (at least 3 chars)
                        elif len(ref_clean) >= 3 and len(val_clean) >= 3:
                            common = 0
                            for a, b in zip(ref_clean, val_clean):
                                if a == b:
                                    common += 1
                                else:
                                    break
                            if common >= min(3, len(ref_clean) - 1):
                                if val not in similar:
                                    similar.append(val)
            except Exception:
                pass

        return similar[:max_results]

    def get_full_content(self, max_rows=100):
        """Get all data from all small tables (for context injection)."""
        all_content = ""
        for fn, info in self.tables.items():
            try:
                rows = self.conn.execute(f'SELECT * FROM "{info["table"]}" LIMIT {max_rows}').fetchall()
                all_content += f"[Excel: {info['nom_affichage']}]\n"
                # Show original column names (with units like €, mm, kg)
                all_content += "En-têtes originaux: " + " | ".join(info['raw_columns']) + "\n"
                all_content += " | ".join(info['columns']) + "\n"
                for row in rows:
                    all_content += " | ".join(str(v) for v in row) + "\n"
                all_content += "\n"
            except Exception:
                pass
        return all_content


# Global SQLite DB for Excel
excel_db = ExcelSQLiteDB()

def mention_amdp(nb):
    if nb >= 2:
        return ("\n\n---\nSi vous souhaitez approfondir votre recherche, contactez AMDP : www.amdp.shop.")
    return ""

def recherche_perplexity(question):
    if not PERPLEXITY_API_KEY:
        return ""
    try:
        headers = {"Authorization": "Bearer " + PERPLEXITY_API_KEY, "Content-Type": "application/json"}
        payload = {"model": "sonar",
                   "messages": [
                       {"role":"system","content":"Expert outillage professionnel. Reponds en francais. Pas de citations [1][2]."},
                       {"role":"user","content":question}
                   ],
                   "max_tokens": 600}
        r = httpx.post("https://api.perplexity.ai/chat/completions", headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("Erreur Perplexity: " + str(e))
        return ""

def generer_question_bienvenue():
    suggestions_bienvenue = [
        "Propose un cliquet ¼ de 72 dents",
        "Quelles clés dynamométriques électroniques sont disponibles ?",
        "Quelles douilles 1/4 métriques sont disponibles ?",
        "Donne-moi les caractéristiques de la référence T72FOD",
    ]
    import random
    return random.choice(suggestions_bienvenue)

# ── Session state — Agent avec mémoire ────────────────────────────────────────
historique_conversation = []
historique_recherches   = {}
MAX_HIST                = 8
derniere_page_trouvee   = None
lastReference_backend   = [None]
# Agent memory
derniers_chunks_trouves = []       # Last RAG chunks found (for follow-up)
dernier_contexte_envoye = ""       # Last context sent to LLM
resume_conversation     = ""       # Running summary of conversation
derniere_question       = ""       # Last user question (for follow-up detection)
derniere_propose_web    = False    # Whether last response proposed web search

# ── Conversation Memory ──
# Simple but effective: track the ACTIVE topic/product, not history of all products
session_memory = {
    "active_ref": None,       # Currently discussed reference (e.g. "T72FOD")
    "active_marque": "",      # Currently discussed brand
    "active_topic": "",       # Current topic summary (e.g. "cliquet Snap-on T72FOD")
    "active_source": "",      # Source catalogue being discussed
    "product_type": "",       # Type of product (cliquet, douille, clé, etc.)
    "recent_refs": [],        # Last 3 references discussed
    "history": [],            # [{question, ref, topic, timestamp}] — last 10 exchanges
}

def detecter_product_type(text):
    """Detect product type from text using keyword matching (FR+EN)."""
    text_low = text.lower()
    type_keywords = {
        "cliquet": ["cliquet", "ratchet", "t72", "t36", "tl72", "thl72"],
        "douille": ["douille", "socket", "sw", "sfs", "tms", "tm", "tsm"],
        "clé": ["clé", "cle", "clef", "wrench", "key", "soex", "flank", "oexm", "goex"],
        "tournevis": ["tournevis", "screwdriver", "sdd", "sgd"],
        "pince": ["pince", "plier", "knipex"],
        "servante": ["servante", "toolbox", "kra", "krsc", "roll cab", "coffre"],
        "embout": ["embout", "bit", "torx", "tamper"],
        "rallonge": ["rallonge", "extension", "fx"],
        "marteau": ["marteau", "hammer", "hbfe", "hbbd"],
        "extracteur": ["extracteur", "extractor", "puller"],
        "clé dynamométrique": ["dynamométrique", "dynamometrique", "torque wrench", "techangle", "atech"],
    }
    for ptype, keywords in type_keywords.items():
        if any(kw in text_low for kw in keywords):
            return ptype
    return ""

def memory_update_active(question, reponse="", reference=None, source=""):
    """Update active topic based on current exchange.
    Key insight: if a new reference or topic appears, REPLACE the active topic.
    Prefix-matching: T72 is a prefix of T72FOD → continuation, keep most specific."""
    q_low = question.lower()

    # Extract references from question + response
    refs = re.findall(r'\b([A-Z]{1,5}\d{2,6}[A-Z0-9]*)\b', question + " " + (reponse or "")[:300])

    # Detect brand
    marque = ""
    for m in ["Snap-on", "Facom", "Totech", "EGA Master", "Bahco", "Beta", "Stahlwille", "Knipex", "Wera", "CAB"]:
        if m.lower() in q_low or m.lower() in (reponse or "").lower()[:300]:
            marque = m
            break

    new_ref = reference or (refs[0] if refs else None)

    # Detect product type from question + response
    ptype = detecter_product_type(question + " " + (reponse or "")[:300])
    if ptype:
        session_memory["product_type"] = ptype

    # Determine if this is a NEW topic or continuation
    is_new_topic = False
    active_ref = session_memory["active_ref"]
    if new_ref and new_ref != active_ref:
        # Check prefix-matching: if one ref is prefix of the other, it's a CONTINUATION
        if active_ref and (active_ref.startswith(new_ref) or new_ref.startswith(active_ref)):
            # Continuation — keep the most specific (longest) reference
            new_ref = new_ref if len(new_ref) >= len(active_ref) else active_ref
            print(f"[MEMORY] Prefix match: continuation, ref={new_ref}")
        else:
            # Truly different reference → new topic
            is_new_topic = True
    elif not new_ref and active_ref:
        # No reference in question — is it a follow-up or new topic?
        active_ref_lower = active_ref.lower()
        question_mentions_active = (
            active_ref_lower in q_low or
            (session_memory["active_marque"] and session_memory["active_marque"].lower() in q_low) or
            (session_memory["product_type"] and session_memory["product_type"] in q_low)
        )
        if not question_mentions_active:
            suivi_patterns = ["c'est", "est-ce", "et la", "et le", "et les", "aussi", "même",
                            "compatible", "version", "oui", "non", "ok", "d'accord", "merci",
                            "quel", "combien", "pourquoi", "comment", "où", "quand",
                            "non fod", "sans fod", "avec fod", "version fod"]
            is_follow_up_pattern = any(p in q_low for p in suivi_patterns)
            is_very_short = len(question.split()) <= 3

            if not is_follow_up_pattern and not is_very_short:
                is_new_topic = True

    if is_new_topic or not active_ref:
        session_memory["active_ref"] = new_ref
        session_memory["active_marque"] = marque
        session_memory["active_topic"] = question[:100]
        if source:
            session_memory["active_source"] = source
        print(f"[MEMORY] Nouveau sujet: ref={new_ref} marque={marque} type={ptype} topic={question[:60]}")
    else:
        # Continuation — update ref if more specific, update marque if detected
        if new_ref and (not active_ref or len(new_ref) > len(active_ref)):
            session_memory["active_ref"] = new_ref
        if marque and not session_memory["active_marque"]:
            session_memory["active_marque"] = marque

    # Update recent_refs (last 3 unique)
    if new_ref:
        rr = session_memory["recent_refs"]
        if new_ref in rr:
            rr.remove(new_ref)
        rr.append(new_ref)
        session_memory["recent_refs"] = rr[-3:]

    # Add to history
    session_memory["history"].append({
        "question": question[:100], "ref": new_ref, "topic": session_memory["active_topic"],
        "time": time.time()
    })
    if len(session_memory["history"]) > 10:
        session_memory["history"] = session_memory["history"][-10:]

def memory_get_active_ref():
    """Get the currently active reference."""
    return session_memory["active_ref"]

def memory_get_active_context():
    """Build context string for the ACTIVE topic only."""
    parts = []
    if session_memory["active_ref"]:
        ref_str = session_memory["active_ref"]
        if session_memory["active_marque"]:
            ref_str += f" ({session_memory['active_marque']})"
        parts.append(f"Produit en cours: {ref_str}")
    if session_memory["product_type"]:
        parts.append(f"Type: {session_memory['product_type']}")
    if session_memory["active_topic"]:
        parts.append(f"Sujet: {session_memory['active_topic']}")
    if session_memory["recent_refs"]:
        parts.append(f"Refs récentes: {', '.join(session_memory['recent_refs'])}")
    return " | ".join(parts) if parts else ""

def detecter_question_suivi(question, derniere_q=""):
    """Detect if the question is a follow-up to the previous exchange."""
    q = question.lower().strip()
    words = q.split()

    # If question contains a product reference (letters+digits), check if it's new
    has_new_ref = bool(re.search(r'[A-Za-z]{1,4}\d{2,6}[A-Za-z0-9]*', question))
    if has_new_ref:
        prev_ref = re.search(r'[A-Za-z]{1,4}\d{2,6}[A-Za-z0-9]*', derniere_q or "")
        new_ref = re.search(r'[A-Za-z]{1,4}\d{2,6}[A-Za-z0-9]*', question)
        if prev_ref and new_ref and prev_ref.group().lower() == new_ref.group().lower():
            return True  # Same reference = follow-up
        return False  # New reference = new topic

    # If question mentions web/internet, it's NOT a follow-up on catalogue
    if any(w in q for w in ["internet", "web", "google", "en ligne"]):
        return False

    # Very short answer (1-2 words) = follow-up
    if len(words) <= 2:
        return True

    # Short question (3-5 words) with pronouns or deictics = follow-up
    pronoms_courts = ["ça", "ca", "il", "elle", "le", "la", "les", "ses", "son", "sa",
                       "ce", "cet", "cette", "ces", "en", "y", "lui", "leur"]
    if len(words) <= 5 and any(w in pronoms_courts for w in words):
        return True

    # Check if new question shares key words with the previous one (same topic)
    if derniere_q:
        prev_words = set(w.lower().strip(".,;:?!()") for w in derniere_q.split() if len(w) >= 4)
        new_words = set(w.lower().strip(".,;:?!()") for w in question.split() if len(w) >= 4)
        common = prev_words & new_words
        if common:
            return True  # Shared topic words = follow-up

    # Pronouns / deictics referencing previous context
    marqueurs_suivi = [
        "et aussi", "et les", "et le", "et la", "pareil pour", "meme chose",
        "celui-ci", "celle-ci", "ce produit", "cet outil", "cette reference",
        "lequel", "laquelle", "lesquels", "lesquelles",
        "en plus", "autre chose", "quoi d'autre", "d'autres",
        "plus de details", "plus d'infos", "explique", "precise",
        "celui la", "celle la", "celui-la", "celle-la",
        "le meme", "la meme", "les memes",
        "quel prix", "quelle taille", "quel poids", "quelle couleur",
        "combien coute", "combien ca coute", "ca fait combien",
        "il fait", "elle fait", "ils font", "elles font",
        "c'est quoi", "ca sert a quoi", "a quoi ca sert",
        "mais avec", "mais en", "mais sur", "plutot", "plutôt",
        "en metrique", "en mm", "en pouces", "traduis", "convertis",
        "est il disponible", "est elle disponible", "est-il", "est-elle",
        "compare", "par rapport",
    ]
    return any(m in q for m in marqueurs_suivi)

def reformuler_question_suivi(question, derniere_q, resume):
    """Use GPT to reformulate a follow-up question with context."""
    try:
        # Use structured memory for context (not cumulative resume)
        active_ctx = memory_get_active_context()
        context_str = active_ctx if active_ctx else (resume[:150] if resume else "")
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=150,
            messages=[
                {"role":"system","content":
                    "Tu recois une question de suivi dans une conversation sur des outils/produits industriels. "
                    "Reformule cette question en une question autonome et complete qui integre le contexte. "
                    "Reponds UNIQUEMENT avec la question reformulee, rien d'autre."},
                {"role":"user","content":
                    f"Contexte actuel: {context_str}\n"
                    f"Question precedente: {derniere_q}\n"
                    f"Question de suivi: {question}\n"
                    f"Question reformulee:"}
            ]
        )
        reformulee = r.choices[0].message.content.strip()
        print(f"[AGENT] Question reformulee: '{question}' -> '{reformulee}'")
        return reformulee
    except Exception as e:
        print(f"Reformulation error: {e}")
        return question

_recent_exchanges = []  # Store last 3 structured exchanges

def mettre_a_jour_resume(question, reponse_courte):
    """Update conversation context — store last 3 exchanges for richer context."""
    global resume_conversation, _recent_exchanges
    ref = extraire_reference_from_text(reponse_courte)
    ptype = detecter_product_type(question + " " + reponse_courte)
    _recent_exchanges.append({
        "q": question[:150],
        "r": reponse_courte[:250],
        "ref": ref or "",
        "type": ptype or "",
    })
    if len(_recent_exchanges) > 3:
        _recent_exchanges = _recent_exchanges[-3:]
    # Build resume from recent exchanges
    parts = []
    for i, ex in enumerate(_recent_exchanges):
        entry = f"Q: {ex['q']} | R: {ex['r']}"
        if ex['ref']:
            entry += f" | Ref: {ex['ref']}"
        if ex['type']:
            entry += f" | Type: {ex['type']}"
        parts.append(entry)
    resume_conversation = " /// ".join(parts)
    print(f"[MEMORY] Resume ({len(_recent_exchanges)} échanges): {resume_conversation[:120]}...")

def clean_markdown(text):
    """Remove ALL markdown formatting from text for clean display."""
    if not text:
        return text
    # Remove headers ### ## #
    text = re.sub(r'#{1,6}\s*', '', text)
    # Remove bold **text** and *text*
    text = re.sub(r'\*{2,3}([^*]+)\*{2,3}', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Remove markdown bullet points
    text = re.sub(r'^\s*[-•]\s+', '• ', text, flags=re.MULTILINE)
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove backticks
    text = text.replace('`', '')
    # Remove double quotes around entire phrases (GPT habit)
    text = re.sub(r'^"(.+)"$', r'\1', text.strip())
    # Remove leftover citation brackets [1] [2]
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()

DISTRIBUTEURS_INTERDITS = [
    "Farnell", "Leroy Merlin", "Conrad", "TME", "Amazon", "Rubix", "Mouser",
    "RS Components", "RS Online", "Cdiscount", "Manomano", "ManoMano", "eBay", "AliExpress",
    "Brico Depot", "Brico Dépôt", "Castorama", "Toolstation", "Wurth", "Würth",
    "Sonepar", "Rexel", "Berner", "Orexad", "Brammer", "DirectIndustry",
    "Mistermenuiserie", "Outillage Online", "123roulement"
]

def clean_distributeurs(text):
    """Remove any mention of competitor distributors from response text."""
    if not text:
        return text
    for dist in DISTRIBUTEURS_INTERDITS:
        # Remove sentences containing distributor names
        pattern = r'[^.]*\b' + re.escape(dist) + r'\b[^.]*\.'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Remove "disponible chez..." patterns
    text = re.sub(r'[Dd]isponibles?\s+chez\s+[^.]+\.', '', text)
    text = re.sub(r'[Vv]ous pouvez (?:le|la|les) trouver (?:chez|sur|à)\s+[^.]+\.', '', text)
    text = re.sub(r'[Ee]n vente (?:chez|sur)\s+[^.]+\.', '', text)
    # Clean up double spaces/newlines
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()

def extraire_reference_from_text(text):
    """Extract product references from text (e.g. T72FOD, SNAP2040, etc.)."""
    refs = re.findall(r'\b([A-Z]{1,5}\d{2,6}[A-Z0-9]*)\b', text or "")
    return refs[0] if refs else ""

def detecter_lien_llm(question, derniere_q, resume):
    """Detect if question is related to previous exchange. Returns (is_linked, question).
    IMPORTANT: Does NOT modify the question. Context is injected via build_llm_messages instead."""
    if not derniere_q and not memory_get_active_ref():
        return False, question
    try:
        # Use structured memory context (not the resume which can be polluted)
        active_ctx = memory_get_active_context()
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=10,
            messages=[
                {"role":"system","content":
                    "Réponds UNIQUEMENT OUI ou NON. "
                    "La nouvelle question fait-elle référence au même sujet/produit que la question PRÉCÉDENTE IMMÉDIATE? "
                    "Ignore les sujets plus anciens."},
                {"role":"user","content":
                    f"Sujet actif: {active_ctx[:100]}\n"
                    f"Question précédente: {derniere_q}\n"
                    f"Nouvelle question: {question}"}
            ]
        )
        resp = r.choices[0].message.content.strip().upper()
        is_linked = "OUI" in resp
        print(f"[AGENT] Lien: {is_linked} | Q: {question[:60]} | Ctx: {active_ctx[:60]}")
        return is_linked, question  # Return UNMODIFIED question
    except Exception as e:
        print(f"[AGENT] Erreur lien: {e}")
        return detecter_question_suivi(question, derniere_q), question

def build_llm_messages(system_prompt, question, contexte, historique, use_previous_context=False):
    """Build LLM messages with conversation history. Memory injection when linked."""
    full_system = system_prompt + "\n\n"
    full_system += (
        "RÈGLE ABSOLUE: ta réponse doit UNIQUEMENT contenir la réponse à la question de l'utilisateur. "
        "Ne JAMAIS mentionner, citer ou répéter le résumé de conversation, le contexte précédent, "
        "les extraits bruts, ou la question reformulée. Ne JAMAIS commencer par 'L'utilisateur s'est renseigné sur...' "
        "ou tout autre récapitulatif. Réponds directement à la question.\n"
        "FORMATAGE: Texte simple et clair. PAS de ### ni ## ni #. PAS de ** ni *. "
        "Termine TOUJOURS tes phrases (ne les coupe jamais en plein milieu). "
        "Ne répète PAS la même information avec des formulations différentes. Sois concis et direct. "
        "PAS de markdown. Utilise des phrases complètes.\n\n"
    )

    # Memory injection — ONLY when question is linked to previous exchange
    # This prevents contaminating new topics with old context
    if use_previous_context:
        # Inject structured memory ONLY (not the full resume which causes topic pollution)
        mem_ctx = memory_get_active_context()
        if mem_ctx:
            full_system += "CONTEXTE ACTUEL (produit/sujet en cours de discussion): " + mem_ctx + "\n\n"
        if dernier_contexte_envoye:
            full_system += (
                "EXTRAITS PRÉCÉDENTS (utiliser si la question est un suivi): "
                + dernier_contexte_envoye[:2000] + "\n\n"
            )

    msgs = [{"role":"system","content":full_system}]

    # Conversation history — limit to recent context only
    # Only include last 4 exchanges to reduce topic contamination
    for h in historique[-4:]:
        msgs.append({"role":h["role"],"content":str(h["content"])[:500]})

    # User message: only the question + new extracts
    msgs.append({"role":"user","content":"Extraits catalogue:\n\n" + contexte + "\n\nQuestion: " + question})
    return msgs

# ── Main question handler ─────────────────────────────────────────────────────
def poser_question(question, session_id="default", mode="catalogue"):
    global derniere_page_trouvee, historique_conversation, historique_recherches, lastReference_backend
    global derniers_chunks_trouves, dernier_contexte_envoye, resume_conversation, derniere_question, derniere_propose_web

    q_low = question.lower()
    question_originale = question  # Keep original for history

    # ── Handle "oui"/"ok" after web proposal → switch to web ──
    if derniere_propose_web and q_low.strip() in ["oui", "ok", "d'accord", "yes", "vas-y", "vasy", "go", "oui merci"]:
        mode = "web"
        # Build a contextual query instead of sending the raw previous question
        ctx = memory_get_active_context()
        if ctx and derniere_question:
            question = reformuler_question_suivi(derniere_question, derniere_question, resume_conversation)
        else:
            question = derniere_question
        q_low = question.lower()

    # ── Agent: detect if question is linked to previous exchange ──
    is_suivi = False
    
    # Use LLM for nuanced detection (only if we have previous context)
    if derniere_question and resume_conversation:
        is_suivi, _ = detecter_lien_llm(question, derniere_question, resume_conversation)
    elif derniere_question:
        # Fallback to simple detection
        is_suivi = detecter_question_suivi(question, derniere_question)

    # Reformulate follow-up questions with context
    if is_suivi and derniere_question:
        question = reformuler_question_suivi(question, derniere_question, resume_conversation)
        q_low = question.lower()

    # Expand vague follow-up using history
    if any(k in q_low for k in ["explorer la gamme","gamme complete","toutes les variantes"]):
        last_user = next((h["content"] for h in reversed(historique_conversation) if h["role"]=="user"), "")
        if last_user and last_user.lower() not in q_low:
            question = "Montre toutes les variantes disponibles de : " + last_user
            q_low    = question.lower()

    if "equivalence" in q_low and len(question) < 35 and lastReference_backend[0]:
        question = "Trouve une equivalence pour la reference " + lastReference_backend[0]
        q_low    = question.lower()

    # Detect explicit web search request → force web mode
    mots_web = [
        "recherche internet","recherche web","recherche google",
        "cherche sur internet","cherche sur le web","cherche sur google",
        "cherche internet","cherche en ligne","cherche online",
        "va sur internet","va sur le web","va sur google",
        "regarde sur internet","regarde sur le web","regarde sur google",
        "regarde en ligne",
        "search online","google",
        "fais une recherche internet","fais une recherche web","fais une recherche en ligne",
        "lance une recherche","fais une recherche",
        "sur internet","sur le web","sur google","en ligne",
        "trouve sur internet","trouve sur le web","trouve en ligne",
        "verifie sur internet","vérifie sur internet",
        "check sur internet","check en ligne",
    ]
    demande_web_explicite = any(m in q_low for m in mots_web)
    # Detect REFUSAL of web search: "non", "pas de recherche", "je ne veux pas"
    refus_web = any(r in q_low for r in [
        "pas de recherche", "non je ne", "non merci", "pas besoin", "sans recherche",
        "ne veux pas", "je ne souhaite pas", "pas la peine", "inutile",
        "pas de web", "pas internet", "reste sur", "non pas", "non,", "non !"
    ])
    if refus_web:
        demande_web_explicite = False
        print(f"[AGENT] Refus web détecté: {question[:60]}")
    if demande_web_explicite and mode != "web":
        mode = "web"
        # Clean the question: remove the web request prefix to keep just the actual query
        prefixes_web = [
            "fais une recherche internet sur ","fais une recherche web sur ",
            "fais une recherche en ligne sur ","fais une recherche sur ",
            "lance une recherche sur ","lance une recherche internet sur ",
            "cherche sur internet ","cherche sur le web ","cherche sur google ",
            "cherche internet ","cherche en ligne ","cherche online ",
            "recherche internet ","recherche web ","recherche google ",
            "va sur internet et ","va sur le web et ","va sur google et ",
            "regarde sur internet ","regarde sur le web ","regarde sur google ",
            "regarde en ligne ","trouve sur internet ","trouve sur le web ",
            "vérifie sur internet ","verifie sur internet ",
        ]
        for prefix in prefixes_web:
            if q_low.startswith(prefix):
                question = question[len(prefix):].strip()
                q_low = question.lower()
                break

    # Voir la page
    mots_voir  = ["voir la page","afficher la page","montre la page","affiche la page",
                  "telecharger la page","voir page","afficher page","affiche cette page",
                  "montre cette page","affiche la page du catalogue","page du catalogue"]
    demande_page = any(m in q_low for m in mots_voir)
    match_page   = re.search(r"page\s*(\d+)", q_low)

    if demande_page:
        page_cible = int(match_page.group(1)) if match_page else derniere_page_trouvee
        if page_cible:
            try:
                img = page_en_image_base64(page_cible)
                historique_conversation.append({"role":"user","content":question})
                historique_conversation.append({"role":"assistant","content":"Page "+str(page_cible)+" affichee."})
                return {"texte":"Page "+str(page_cible)+" du catalogue.",
                        "image":img, "sources":[CATALOGUE_NOM_PRINCIPAL+".pdf"],
                        "fiabilite":100,
                        "suggestions":["Voir les références de cette page",
                                       "Chercher un produit similaire"],
                        "page_num":page_cible, "page_nums":[page_cible] if page_cible else [], "from_excel":False, "propose_web":False}
            except Exception as e:
                print("Page image error:", e)

    mots    = q_low.split()
    q_lower = q_low

    # Detect specific supp PDF by name
    nom_supp_vise = None
    for nom, data in pdfs_supplementaires.items():
        nom_court = nom.lower().replace(".pdf","").replace("_"," ").replace("-"," ")
        if any(m in q_lower for m in [w for w in nom_court.split() if len(w) > 2]):
            nom_supp_vise = nom
            break

    # Meta question
    if (any(m in mots for m in ["pages","taille","indexe","indexees"])
            and not nom_supp_vise and "contient" in mots):
        nb   = len(set(p["page"] for p in vector_db.ids_main))
        supp = (" + "+str(len(pdfs_supplementaires))+" doc(s)") if pdfs_supplementaires else ""
        return {"texte":"Catalogue principal: "+str(nb)+" pages, "+str(len(vector_db.ids_main))+" extraits."+supp,
                "image":None,"sources":["Catalogue principal"],"fiabilite":100,
                "suggestions":[],"page_num":None,"page_nums":[],"from_excel":False,"propose_web":False}

    mots_image    = ["image","photo","montre","affiche","visuel","illustration","apercu","voir","picture"]
    demande_image_mots = any(m in mots for m in mots_image)
    demande_image = demande_image_mots and mode != "web" and not any(w in q_low for w in ["web","internet","en ligne"])
    
    # If asking for image AND it's a follow-up → auto-switch to web mode with context
    if demande_image_mots and not any(w in q_low for w in ["catalogue","page"]):
        # Check if we have a reference from structured memory first
        prev_ref = None
        if memory_get_active_ref():
            prev_ref = memory_get_active_ref()
        if not prev_ref and is_suivi:
            prev_ref = extraire_reference_from_text(resume_conversation or derniere_question or "")
        if not prev_ref and lastReference_backend[0]:
            prev_ref = lastReference_backend[0]
        if not prev_ref:
            prev_ref = extraire_reference_from_text(derniere_question or "")
        
        if prev_ref and not re.search(r'[A-Z]{1,5}\d{2,6}', question.upper()):
            # User asks for image without specifying a ref → use previous ref
            marque_ctx = ""
            if memory_get_active_ref():
                marque_ctx = session_memory.get("active_marque", "")
            mode = "web"
            question = f"Image du produit {marque_ctx} {prev_ref}".strip()
            q_low = question.lower()
            demande_image = False
            print(f"[AGENT] Image → web avec ref: {question}")

    reference = detecter_reference(question)
    specs     = extraire_specs(question)
    large     = any(m in mots for m in MOTS_LARGE)

    # Update active topic EARLY so memory is correct for this exchange
    memory_update_active(question, "", reference)

    cle = reference if reference else " ".join(sorted([m for m in mots if len(m) > 4]))[:50]
    historique_recherches[cle] = historique_recherches.get(cle, 0) + 1
    nb_similaires = historique_recherches[cle]

    # ── Specific supp PDF ──
    if nom_supp_vise:
        query_en = traduire_query(question)
        # Use vector search on supplementary PDF
        results = vector_db.search_supplementary(nom_supp_vise, question, query_en, n=10)
        nom_affichage_supp = pdfs_supplementaires[nom_supp_vise]["nom_affichage"]

        if results:
            # Rerank
            results = reranker_gpt(question, results, top_n=6, query_en=query_en)
            ctx = "\n\n".join(["[Page "+str(r.get("page","?"))+"]\n"+r["texte"] for r in results])
            # Extract page numbers for download buttons
            supp_pages = list(set([r.get("page") for r in results if r.get("page")]))
            supp_pages.sort()
            supp_page_num = supp_pages[0] if supp_pages else None
            r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=1500,
                messages=[{"role":"system","content":
                    "Expert outillage professionnel. Reponds en francais. Cite les pages sources. "
                    "Texte simple sans markdown (pas de ###, **, etc). "
                    "INTERDIT: citer des distributeurs concurrents (Farnell, Leroy Merlin, Conrad, TME, Amazon, Rubix, Mouser, RS Components, Cdiscount, Manomano, eBay, AliExpress, Brico Depot, Castorama). "
                    "Le seul site autorise est amdp.shop."},
                          {"role":"user","content":"Extraits de "+nom_affichage_supp+":\n\n"+ctx+"\n\nQuestion: "+question}])
            reponse_supp = clean_distributeurs(clean_markdown(r.choices[0].message.content)) + mention_amdp(nb_similaires)
            return {"texte":reponse_supp,
                    "image":None,"sources":[nom_affichage_supp+".pdf"],
                    "fiabilite":85,
                    "suggestions":[],"page_num":supp_page_num,"page_nums":supp_pages,
                    "from_excel":False,"propose_web":False}
        else:
            # Fallback: text search on raw content
            texte_supp = pdfs_supplementaires[nom_supp_vise]["texte"]
            apercu = texte_supp[:3000]
            r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=800,
                messages=[{"role":"system","content":"Expert outillage. Reponds en francais."},
                          {"role":"user","content":"Document "+nom_affichage_supp+":\n\n"+apercu+"\n\nQuestion: "+question}])
            return {"texte":r.choices[0].message.content+mention_amdp(nb_similaires),
                    "image":None,"sources":[nom_affichage_supp+".pdf"],"fiabilite":70,
                    "suggestions":[],"page_num":None,"page_nums":[],"from_excel":False,"propose_web":False}

    # ── Main catalogue search with RAG ──
    # When it's a follow-up with a vague query, enrich the SEARCH query (not the LLM query)
    search_question = question
    if is_suivi and memory_get_active_ref():
        # Check if question is vague (no reference, short, or uses pronouns)
        has_ref_in_q = bool(re.search(r'\b[A-Z]{1,5}\d{2,6}[A-Z0-9]*\b', question))
        if not has_ref_in_q:
            active_ref = memory_get_active_ref()
            active_marque = session_memory.get("active_marque", "")
            search_question = f"{question} {active_ref} {active_marque}".strip()
            print(f"[RAG] Enriched search: {search_question[:80]}")
    
    query_en = traduire_query(search_question)
    n_chunks = 20 if large else 15  # Fetch more, then rerank

    pertinents_catalogue = vector_db.search_main(question, query_en, n=n_chunks, reference=reference, specs=specs)

    # Rerank to get the best chunks
    if pertinents_catalogue:
        pertinents_catalogue = reranker_gpt(question, pertinents_catalogue, top_n=10 if large else 8, query_en=query_en)

    # Search supplementary PDFs too (for "both" mode)
    contexte_supp = ""
    supp_results = vector_db.search_all_supplementary(question, query_en, n=3)
    if supp_results:
        contexte_supp = "\n\n".join(
            ["[Document: "+r.get("_source_file","?")+", Page "+str(r.get("page","?"))+"]\n"+r["texte"]
             for r in supp_results]
        )

    # Excel search (unchanged logic)
    # Excel search — use SQLite Text-to-SQL if available, fallback to text search
    contexte_excel          = ""
    excel_sources_utilisees = []
    if excel_db.tables:
        # Try SQL query first
        sql_result, sql_sources = excel_db.query_with_llm(question, reference)
        if sql_result:
            contexte_excel = sql_result
            for s in sql_sources:
                ext = "xlsx"
                excel_sources_utilisees.append(s["nom_affichage"] + "." + ext)
        else:
            # Fallback: get all content from small tables
            contexte_excel = excel_db.get_full_content(max_rows=50)
            if contexte_excel:
                for fn, info in excel_db.tables.items():
                    excel_sources_utilisees.append(info["nom_affichage"] + ".xlsx")
    elif excels_supplementaires:
        # Legacy fallback if SQLite not populated
        mots_q = [m.strip(".,;:?!()").lower() for m in question.split() if len(m) >= 2]
        if reference:
            mots_q.append(reference.lower())
        for nom_fichier, data in excels_supplementaires.items():
            lignes = data["texte"].split("\n")
            if len(lignes) <= 50:
                contexte_excel += "[Excel: "+data["nom_affichage"]+"]\n"+"\n".join(lignes)+"\n\n"
                ext = nom_fichier.rsplit(".",1)[-1].lower() if "." in nom_fichier else "xlsx"
                excel_sources_utilisees.append(data["nom_affichage"]+"."+ext)
            else:
                header_lines = lignes[:3]
                pertinentes = [l for l in lignes if any(m in l.lower() for m in mots_q)]
                if pertinentes:
                    contexte_excel += "[Excel: "+data["nom_affichage"]+"]\n"
                    contexte_excel += "\n".join(header_lines) + "\n---\n"
                    contexte_excel += "\n".join(pertinentes[:50])+"\n\n"
                    ext = nom_fichier.rsplit(".",1)[-1].lower() if "." in nom_fichier else "xlsx"
                    excel_sources_utilisees.append(data["nom_affichage"]+"."+ext)
    if contexte_excel:
        contexte_supp = (contexte_supp+"\n\n"+contexte_excel).strip()

    # System prompts
    system_web = (
        "Tu es expert en outillage professionnel. Reponds en francais. "
        "Tu peux mentionner d autres marques si demande. "
        "REGLE ABSOLUE: Ne JAMAIS citer de distributeurs ou revendeurs. "
        "Liste noire: Farnell, Leroy Merlin, Conrad, TME, Amazon, Rubix, Mouser, RS Components, "
        "Cdiscount, Manomano, eBay, AliExpress, Brico Depot, Castorama, Toolstation, Wurth, Sonepar, "
        "Rexel, Berner, Orexad, Brammer, DirectIndustry. "
        "Ne dis JAMAIS 'disponible chez' suivi d'un nom de distributeur. "
        "Le seul site autorise est amdp.shop. Texte simple sans markdown (pas de ### ni **). "
        "Termine par 1 suggestion:\nSUGG:..."
    )
    system_excel = (
        "Tu es expert en analyse de données outillage. Réponds en français. "
        "RÈGLE ABSOLUE: retranscris les valeurs EXACTES trouvées dans les données Excel fournies. "
        "Ne JAMAIS inventer, estimer ou deviner un prix, une référence ou une valeur. "
        "IMPORTANT: les références produit peuvent avoir des variantes d'écriture (espaces, tirets). "
        "Par exemple 'T72FOD', 'T72 FOD', 'TL72FOD', 'TL72 FOD' désignent probablement le même produit. "
        "Si la référence demandée ressemble à une référence du fichier (même en ignorant espaces/tirets), considère que c'est la même. "
        "IMPORTANT: quand tu cites un prix ou une mesure, TOUJOURS préciser l'unité (€, $, mm, kg, etc). "
        "Regarde les en-têtes de colonnes pour trouver l'unité (ex: 'Prix (€)' ou 'Prix de vente' avec des valeurs en euros). "
        "Si la donnée demandée est présente dans les extraits, cite-la telle quelle avec son unité. "
        "Si la donnée n'est PAS dans les extraits, dis clairement que tu ne l'as pas trouvée. "
        "Texte simple sans markdown. "
        "Termine par 1 suggestion:\nSUGG:Afficher toutes les références"
    )
    system_catalogue = (
        "Tu es un assistant expert en outillage professionnel pour AMDP, distributeur multi-marques. "
        "Tu analyses des catalogues de différentes marques (Snap-on, Facom, Totech, etc). "
        "Réponds UNIQUEMENT en français avec les accents. "
        "RÈGLE CRITIQUE: Réponds UNIQUEMENT en te basant sur les EXTRAITS FOURNIS dans le message utilisateur. "
        "Ne mélange JAMAIS les informations de produits/catalogues différents. "
        "Si les extraits ne contiennent pas l'information demandée, dis-le clairement. "
        "Ne complète JAMAIS avec des informations de conversations précédentes sauf si la question y fait explicitement référence. "
        "Analyse les extraits du catalogue pour extraire caractéristiques, tableaux, dimensions. "
        "OBLIGATOIRE: citer le numéro de page source (ex: page 330). "
        "INTERDIT: utiliser les lettres A B C D pour nommer les dimensions. "
        "OBLIGATOIRE: noms réels: largeur, longueur, diamètre, hauteur. "
        "INTERDIT: convertir les unités (pouces, inches, etc) en millimètres ou en métrique sauf si demandé. "
        "Garde les unités TELLES QUE dans le catalogue. "
        "Ne commence JAMAIS par un point ou phrase négative. "
        "INTERDIT: dire que tu ne peux pas accéder aux catalogues ou à internet. Tu as accès aux extraits fournis. "
        "INTERDIT: citer des sites internet de distributeurs concurrents (Rubix, Farnell, Mouser, Amazon, RS Components, Cdiscount, Manomano, etc). "
        "INTERDIT: sites internet autres que amdp.shop. "
        "Si la question est vague: pose UNE question de clarification courte. "
        "Si aucun extrait ne concerne la question: réponds AUCUN_RESULTAT + question clarification. "
        "Texte simple SANS markdown (pas de **, ##, [1]). "
        "Termine OBLIGATOIREMENT par:\n"
        "SUGG:une suggestion pertinente"
    )

    if mode == "web":
        system = system_web
    elif mode == "excel":
        system = system_excel
    else:
        system = system_catalogue

    sources_utilisees = []
    reponse           = ""
    from_excel_only   = False
    page_num_result   = None
    page_nums_result  = []

    # ── Mode excel ────────────────────────────────────────────────────────────
    if mode == "excel":
        if not excels_supplementaires and not excel_db.tables:
            return {"texte":"Aucun fichier Excel chargé. Ajoutez un fichier via la sidebar.",
                    "image":None,"sources":[],"fiabilite":0,
                    "suggestions":["Passer en mode Catalogue"],
                    "page_num":None,"from_excel":True,"propose_web":False}

        # Strategy: always provide full content of small files + SQL results if available
        excel_context_parts = []

        # 1. Try SQL query for targeted results
        if excel_db.tables:
            sql_result, sql_sources = excel_db.query_with_llm(question, reference)
            if sql_result:
                excel_context_parts.append("RESULTATS SQL:\n" + sql_result)
                for s in sql_sources:
                    if s["nom_affichage"] + ".xlsx" not in excel_sources_utilisees:
                        excel_sources_utilisees.append(s["nom_affichage"] + ".xlsx")

        # 2. Always add full content — try SQLite first, then raw text as fallback
        full_added = False
        if excel_db.tables:
            full_content = excel_db.get_full_content(max_rows=200)
            if full_content and len(full_content.strip()) > 20:
                excel_context_parts.append("CONTENU COMPLET DES FICHIERS:\n" + full_content)
                for fn, info in excel_db.tables.items():
                    if info["nom_affichage"] + ".xlsx" not in excel_sources_utilisees:
                        excel_sources_utilisees.append(info["nom_affichage"] + ".xlsx")
                full_added = True

        # Always include raw text as ultimate fallback
        if excels_supplementaires:
            for nom_fichier, data in excels_supplementaires.items():
                if not full_added:
                    excel_context_parts.append("[Excel brut: " + data["nom_affichage"] + "]\n" + data["texte"])
                    ext = nom_fichier.rsplit(".",1)[-1].lower() if "." in nom_fichier else "xlsx"
                    if data["nom_affichage"] + "." + ext not in excel_sources_utilisees:
                        excel_sources_utilisees.append(data["nom_affichage"] + "." + ext)

        contexte_excel = "\n\n".join(excel_context_parts)

        # Add explicit instruction for fuzzy matching when content is small
        if contexte_excel and len(contexte_excel) < 5000:
            contexte_excel = (
                "INSTRUCTION IMPORTANTE: Voici le contenu COMPLET du fichier Excel. "
                "Tu DOIS trouver la réponse dans ces données. "
                "ATTENTION: les références dans le fichier peuvent différer légèrement de la question "
                "(espaces, tirets, préfixes supplémentaires). Exemples de correspondances: "
                "'T72FOD' = 'TL72 FOD' = 'T72-FOD' = 'TL72FOD'. "
                "Ignore les différences d'espaces, tirets et préfixes pour trouver le produit. "
                "Si une ligne contient des caractères similaires à la référence demandée, c'est probablement le bon produit.\n\n"
                + contexte_excel
            )
        print(f"[EXCEL MODE] context length: {len(contexte_excel)}, sources: {excel_sources_utilisees}")
        print(f"[EXCEL MODE] tables in SQLite: {list(excel_db.tables.keys())}")
        print(f"[EXCEL MODE] excels_supplementaires: {list(excels_supplementaires.keys())}")
        if contexte_excel:
            print(f"[EXCEL MODE] first 200 chars: {contexte_excel[:200]}")
        sources_utilisees = list(excel_sources_utilisees) if excel_sources_utilisees else []

        if contexte_excel:
            hist_msgs = build_llm_messages(system, question, contexte_excel, historique_conversation, is_suivi)
            r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=2500, messages=hist_msgs)
            reponse = r.choices[0].message.content

            # If LLM says not found, suggest similar references
            not_found_markers = ["pas trouvé", "pas présent", "n'existe pas", "n'ai pas trouvé", "pas dans le fichier", "aucune donnée", "pas disponible"]
            if any(m in reponse.lower() for m in not_found_markers) and reference:
                similar = excel_db.find_similar_refs(reference)
                if similar:
                    reponse += "\n\nRéférences similaires trouvées dans le fichier :\n"
                    for s in similar:
                        reponse += f"- {s}\n"
                    reponse += "\nVoulez-vous le prix d'une de ces références ?"
        else:
            # No context at all — still try to suggest similar refs
            if reference and excel_db.tables:
                similar = excel_db.find_similar_refs(reference)
                if similar:
                    reponse = f"La référence {reference} n'a pas été trouvée dans les fichiers Excel.\n\n"
                    reponse += "Références similaires trouvées :\n"
                    for s in similar:
                        reponse += f"- {s}\n"
                    reponse += "\nVoulez-vous le prix d'une de ces références ?"
                else:
                    reponse = "Aucune donnée trouvée dans les fichiers Excel pour cette question."
            else:
                reponse = "Aucune donnée trouvée dans les fichiers Excel pour cette question."
        from_excel_only = True

    # ── Mode web ──────────────────────────────────────────────────────────────
    elif mode == "web":
        # Build web query — ALWAYS enrich with active context
        web_query = question
        mem_ctx = memory_get_active_context()
        if mem_ctx:
            web_query = question + " (contexte: " + mem_ctx + ")"
        elif is_suivi and resume_conversation:
            web_query = question + " (contexte: " + resume_conversation[:150] + ")"
        elif len(question.split()) < 8 and dernier_contexte_envoye:
            last_ref = re.search(r'\b([A-Z]{1,4}\d{2,6}[A-Z0-9]*)\b', dernier_contexte_envoye[:500])
            if last_ref:
                web_query = question + " " + last_ref.group(1)
        web = recherche_perplexity(web_query)
        if web:
            reponse           = clean_markdown(web)
            sources_utilisees = ["Recherche web"]
        else:
            reponse = "Aucun résultat web. Vérifiez la clé Perplexity."

    # ── Mode catalogue / both ─────────────────────────────────────────────────
    else:
        if pertinents_catalogue:
            contexte = "\n\n".join(["[Page "+str(p["page"])+"]\n"+p["texte"] for p in pertinents_catalogue])
            if contexte_supp:
                contexte = contexte+"\n\n"+contexte_supp
                sources_utilisees.append(CATALOGUE_NOM_PRINCIPAL+".pdf")
                # Only add supplementary sources that actually have results (not ALL pdfs)
                if supp_results:
                    supp_source_files = set(r.get("_source_file", "") for r in supp_results if r.get("_source_file"))
                    for sf in supp_source_files:
                        if sf in pdfs_supplementaires:
                            sources_utilisees.append(pdfs_supplementaires[sf]["nom_affichage"]+".pdf")
            else:
                sources_utilisees.append(CATALOGUE_NOM_PRINCIPAL+".pdf")

            # Save context for follow-up
            dernier_contexte_envoye = contexte[:3000]

            if large:
                hist_msgs = build_llm_messages(system, question, contexte, historique_conversation, is_suivi)
                r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=2500, messages=hist_msgs)
            else:
                pages_vues = []
                for p in pertinents_catalogue:
                    if p["page"] not in pages_vues: pages_vues.append(p["page"])
                    if len(pages_vues) >= 3: break

                # System prompt enriched with memory (hidden)
                vision_system = system + "\n\n"
                vision_system += (
                    "REGLE ABSOLUE: reponds UNIQUEMENT a la question. "
                    "Ne JAMAIS reciter le resume, le contexte precedent ou les extraits bruts. "
                    "Reponds directement.\n\n"
                )
                if is_suivi:
                    mem_ctx = memory_get_active_context()
                    if mem_ctx:
                        vision_system += "CONTEXTE ACTUEL: " + mem_ctx + "\n\n"
                if is_suivi and dernier_contexte_envoye:
                    vision_system += "EXTRAITS PRÉCÉDENTS: " + dernier_contexte_envoye[:1500] + "\n\n"

                text_part = "Extraits catalogue:\n\n" + contexte + "\n\nQuestion: " + question

                contenu = [{"type":"text","text":text_part}]
                for num in pages_vues:
                    try:
                        img = page_en_image_base64(num)
                        contenu.append({"type":"text","text":"--- Page "+str(num)+" ---"})
                        contenu.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+img,"detail":"low"}})
                    except Exception:
                        pass

                hist_msgs_base = [{"role":"system","content":vision_system}]
                for h in historique_conversation[-6:]:
                    hist_msgs_base.append({"role":h["role"],"content":str(h["content"])})
                msgs_vision = hist_msgs_base[:]
                msgs_vision.append({"role":"user","content":contenu})
                try:
                    r = client.chat.completions.create(model="gpt-4o", max_tokens=2500, messages=msgs_vision)
                except Exception:
                    hist_msgs = build_llm_messages(system, question, contexte, historique_conversation, is_suivi)
                    r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=2500, messages=hist_msgs)
            reponse = r.choices[0].message.content

        elif contexte_supp:
            if excel_sources_utilisees:
                sources_utilisees.extend(excel_sources_utilisees)
                from_excel_only = True
            else:
                for n2,d2 in pdfs_supplementaires.items():
                    sources_utilisees.append(d2["nom_affichage"]+".pdf")
            hist_msgs = build_llm_messages(system, question, contexte_supp, historique_conversation, is_suivi)
            r = client.chat.completions.create(model="gpt-4o-mini", max_tokens=2500, messages=hist_msgs)
            reponse = r.choices[0].message.content

    # Clean — detect "no result" markers BEFORE stripping them
    non_trouve = bool(re.search(r'(?i)a[un]*cun[_\s]*r[eé]sultat', reponse))
    # Also detect when LLM says it didn't find something
    if not non_trouve and mode == "catalogue":
        markers_not_found = [
            "pas trouvé", "pas trouvée", "n'ai pas trouvé",
            "ne figure pas", "n'apparaît pas", "n'est pas mentionné",
            "pas dans le catalogue", "pas dans les catalogues",
            "pas d'information", "aucune information",
            "je ne peux pas", "impossible de trouver"
        ]
        if any(m in reponse.lower() for m in markers_not_found):
            non_trouve = True

    # Remove AUCUN_RESULTAT and all its variants (with accents, spaces, underscores)
    reponse = re.sub(r'(?i)a[un]*cun[_\s]*r[eé]sultat\s*\+?\s*', '', reponse).strip()
    reponse = reponse.replace("RECHERCHE_COMPLEMENTAIRE_POSSIBLE","").strip()
    # Remove "question clarification" literal text that GPT sometimes outputs
    reponse = re.sub(r'(?i)\+?\s*question\s+(?:de\s+)?clarification\s*:?\s*', '', reponse).strip()
    # Clean leading punctuation after removal
    reponse = reponse.lstrip("+ .-").strip()
    reponse = clean_markdown(reponse).lstrip(". ").strip()
    reponse = clean_distributeurs(reponse)

    # No result in catalogue -> keep partial answer but propose web
    if non_trouve and mode == "catalogue":
        derniere_propose_web = True
        # Save state before returning
        historique_conversation.append({"role":"user","content":question_originale})
        historique_conversation.append({"role":"assistant","content":reponse[:200] if reponse else "Non trouvé"})
        if len(historique_conversation) > MAX_HIST * 2:
            historique_conversation.pop(0); historique_conversation.pop(0)
        derniere_question = question_originale
        # Update resume so web search has context
        try:
            mettre_a_jour_resume(question_originale, reponse[:200] if reponse else "Non trouvé dans les catalogues")
        except Exception:
            pass

        if reponse and len(reponse) > 30:
            reponse += "\n\nSouhaitez-vous une recherche web ?"
            return {"texte":reponse,"image":None,
                    "sources":sources_utilisees or [CATALOGUE_NOM_PRINCIPAL+".pdf"],"fiabilite":30,
                    "suggestions":["Rechercher sur le web"],
                    "page_num":page_num_result,"page_nums":page_nums_result,
                    "from_excel":False,"propose_web":True,"mode_used":"catalogue"}
        else:
            reponse = "Je n'ai pas trouvé cette information dans les catalogues. Souhaitez-vous une recherche web ?"
            return {"texte":reponse,"image":None,
                    "sources":[CATALOGUE_NOM_PRINCIPAL+".pdf"],"fiabilite":0,
                    "suggestions":["Rechercher sur le web"],
                    "page_num":None,"page_nums":[],
                    "from_excel":False,"propose_web":True,"mode_used":"catalogue"}

    if not reponse:
        reponse = "Information non trouvée dans les catalogues."
    if not sources_utilisees:
        sources_utilisees = ["Catalogue"]

    # AMDP link — only for catalogue/web, NOT excel, only if reference found in catalogue results
    if reference and len(reference) >= 4 and not from_excel_only and not reference[-2:].lower() in ["nm","mm","cm","kg","lb","oz","ft"]:
        # Only show link if reference was actually found in catalogue chunks (not hallucinated)
        ref_in_context = pertinents_catalogue and any(reference.lower() in p.get("texte","").lower() for p in pertinents_catalogue)
        if ref_in_context:
            reponse += "\n\nFiche produit : https://amdp.shop/recherche?s=" + reference

    # Extract SUGG — handle all variants: "SUGG:", "SUGG :", "Sugg:", "Suggestion:",
    # "SUGG\n", "SUGG : ...", etc.
    parts          = re.split(r'(?i)\n*\s*SUGG(?:ESTION)?\s*:?\s*', reponse)
    texte_propre   = parts[0].strip().rstrip("\n")
    suggestions_sugg = []
    for p in parts[1:]:
        s = p.strip().split("\n")[0].strip().rstrip(".")
        if s and len(s) > 3 and len(s) < 120:
            suggestions_sugg.append(s)
    suggestions_sugg = suggestions_sugg[:1]
    reponse = texte_propre
    # Final cleanup: remove any leftover "SUGG" text that wasn't caught by split
    reponse = re.sub(r'(?i)\bSUGG(?:ESTION)?\s*:?\s*$', '', reponse).strip()

    # Fiabilite
    if from_excel_only:
        fiabilite = 90
    elif pertinents_catalogue:
        fiabilite = 85 if len(pertinents_catalogue) < 5 else 95
    else:
        fiabilite = 60
    if "Recherche web" in " ".join(sources_utilisees):
        fiabilite = min(fiabilite, 70)

    # Image
    image_b64 = None
    if demande_image and pertinents_catalogue and not from_excel_only:
        try:
            image_b64 = page_en_image_base64(pertinents_catalogue[0]["page"])
        except Exception:
            pass

    if not from_excel_only:
        reponse += mention_amdp(nb_similaires)

    # Save state — extract ALL page numbers from LLM response
    page_nums_result = []  # List of all pages cited
    # Extract page numbers ONLY from the LLM response text (not from raw chunks)
    if not from_excel_only and reponse:
        page_mentions = re.findall(r'page\s*(\d+)', reponse.lower())
        if page_mentions:
            seen_pages = set()
            for pm in page_mentions:
                p = int(pm)
                if p not in seen_pages and p > 0:
                    page_nums_result.append(p)
                    seen_pages.add(p)
            page_num_result = page_nums_result[0] if page_nums_result else None
        # Don't auto-add pages from chunks when LLM didn't cite any —
        # this caused irrelevant page buttons to appear
        if page_num_result:
            derniere_page_trouvee = page_num_result
    ref_match = re.search(r'\b([A-Z]{1,4}[0-9]{2,6}[A-Z0-9]*)\b', reponse)
    if ref_match:
        lastReference_backend[0] = ref_match.group(1)

    historique_conversation.append({"role":"user","content":question_originale})
    historique_conversation.append({"role":"assistant","content":reponse[:800]})
    if len(historique_conversation) > MAX_HIST * 2:
        historique_conversation.pop(0); historique_conversation.pop(0)

    # Agent memory: save chunks and update summary
    if pertinents_catalogue:
        derniers_chunks_trouves = pertinents_catalogue
    derniere_question = question_originale
    derniere_propose_web = False  # Reset by default

    # Structured memory: update active topic based on this exchange
    detected_ref = reference if reference else None
    active_source = sources_utilisees[0] if sources_utilisees else ""
    memory_update_active(question_originale, reponse[:500], detected_ref, active_source)

    # Update conversation summary (async-friendly, runs in background)
    try:
        mettre_a_jour_resume(question_originale, reponse[:300])
    except Exception:
        pass

    return {"texte":reponse,"image":image_b64,"sources":sources_utilisees,
            "fiabilite":fiabilite,"suggestions":suggestions_sugg,
            "page_num":None if from_excel_only else page_num_result,
            "page_nums":[] if from_excel_only else page_nums_result,
            "from_excel":from_excel_only,"propose_web":False,
            "mode_used":mode}

def generer_suggestions():
    base = [
        ("Suggestion", "Découvrir les servantes Snap-on"),
        ("Equivalence", "Trouve l'équivalence du cliquet T72"),
        ("Gamme", "Quelles clés dynamométriques électroniques ?"),
    ]
    for nom, data in list(pdfs_supplementaires.items())[:1]:
        na = data["nom_affichage"][:25]
        base.append(("Document", "Que contient le document " + na + " ?"))
    return base

# ── Charger fichiers au demarrage ─────────────────────────────────────────────
def charger_fichiers():
    print("[BOOT] Chargement des fichiers...", flush=True)
    if not os.path.exists(PDF_LOCAL):
        print(f"[BOOT] Téléchargement catalogue...", flush=True)
        telecharger_drive(DRIVE_CATALOGUE,  PDF_LOCAL)
    else:
        print(f"[BOOT] Catalogue déjà présent", flush=True)
    if not os.path.exists(INDEX_LOCAL):
        print(f"[BOOT] Téléchargement index...", flush=True)
        telecharger_drive(DRIVE_INDEX,       INDEX_LOCAL)
    else:
        print(f"[BOOT] Index déjà présent", flush=True)
    if not os.path.exists(EMBEDDINGS_LOCAL):
        print(f"[BOOT] Téléchargement embeddings...", flush=True)
        telecharger_drive(DRIVE_EMBEDDINGS,  EMBEDDINGS_LOCAL)
    else:
        print(f"[BOOT] Embeddings déjà présents", flush=True)
    # Download fiche produit template if Drive ID is set
    if DRIVE_FICHE_TEMPLATE and not os.path.exists(FICHE_TEMPLATE):
        try:
            telecharger_drive(DRIVE_FICHE_TEMPLATE, FICHE_TEMPLATE)
            print(f"[BOOT] Template fiche produit téléchargé: {FICHE_TEMPLATE}")
        except Exception as e:
            print(f"[BOOT] Template fiche produit non disponible: {e}")

    # Copy logo if available
    logo_src = os.path.join(BASE_DIR, "Logo_AMDP_2022.png")
    logo_dst = os.path.join(BASE_DIR, "logo_amdp.png")
    if not os.path.exists(logo_dst):
        if os.path.exists(logo_src):
            import shutil; shutil.copy2(logo_src, logo_dst)
        else:
            # Create a simple placeholder
            try:
                doc = fitz.open()
                p = doc.new_page(width=200, height=80)
                p.insert_text(fitz.Point(10, 50), "AMDP", fontsize=40, fontname="helv", color=(0.106, 0.165, 0.290))
                doc.save(logo_dst.replace('.png', '.pdf'))
                doc.close()
            except Exception:
                pass

    # Load index
    with open(INDEX_LOCAL, "r", encoding="utf-8") as f:
        ids = json.load(f)

    # Load embeddings (.npy format)
    if os.path.exists(EMBEDDINGS_LOCAL):
        legacy_embeddings = np.load(EMBEDDINGS_LOCAL).astype(np.float32)
        print(f"[BOOT] Loaded embeddings: {legacy_embeddings.shape}")
    else:
        print("[BOOT] WARNING: No embeddings file found!")
        legacy_embeddings = None

    # Verify dimensions match
    if legacy_embeddings is not None and len(ids) != legacy_embeddings.shape[0]:
        print(f"[BOOT] WARNING: Index ({len(ids)} chunks) != Embeddings ({legacy_embeddings.shape[0]})")
        print("[BOOT] Using index without embeddings — run reindex.py to fix")
        legacy_embeddings = None

    print(f"Loaded {len(ids)} chunks" + (f", embeddings shape: {legacy_embeddings.shape}" if legacy_embeddings is not None else " (no embeddings)"))

    # Index in vector DB
    vector_db.index_main_catalogue(ids, legacy_embeddings)

    return ids

# ══════════════════════════════════════════════════════════════════════════════
# HTML PAGE (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" type="image/png" href="data:image/png;base64,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"/>
<title>AMDP — Catalogue IA</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--orange:#e8501a;--orange2:#f26522;--dark:#1a1a1a;--gray:#f4f4f4;--gray2:#e8e8e8;--white:#fff;--text:#333;--muted:#888;--border:#e0e0e0}
body{font-family:'Nunito','Century Gothic',Arial,sans-serif;background:var(--gray);color:var(--text);height:100vh;display:flex;flex-direction:column;overflow:hidden}
/* LOGIN */
#loginScreen{position:fixed;inset:0;background:var(--dark);display:flex;align-items:center;justify-content:center;z-index:100}
.lbox{background:var(--white);border-radius:6px;padding:40px 36px;width:340px;display:flex;flex-direction:column;gap:14px;box-shadow:0 4px 24px rgba(0,0,0,0.3)}
.lbox-logo{display:flex;flex-direction:column;align-items:center;margin-bottom:8px}
.lbox-logo .logo-name{font-size:32px;font-weight:900;color:var(--orange);letter-spacing:2px}
.lbox-logo .logo-sub{font-size:10px;font-weight:700;color:var(--muted);letter-spacing:3px;text-transform:uppercase}
.lbox input{background:var(--white);border:1px solid var(--border);border-radius:4px;padding:11px 14px;color:var(--text);font-size:14px;font-family:inherit;outline:none;width:100%;transition:border-color 0.2s}
.lbox input:focus{border-color:var(--orange)}
.lbox input::placeholder{color:var(--muted)}
.lbtn{background:var(--orange);border:none;border-radius:4px;color:white;padding:12px;font-size:14px;font-weight:700;font-family:inherit;cursor:pointer;width:100%;margin-top:4px;text-transform:uppercase;letter-spacing:1px;transition:background 0.2s}
.lbtn:hover{background:var(--orange2)}
.lerr{color:#c0392b;font-size:13px;text-align:center;display:none}
/* APP */
#appScreen{display:none;flex-direction:row;height:100vh;overflow:hidden}
/* SIDEBAR */
.side{width:240px;background:var(--white);border-right:1px solid var(--border);flex-shrink:0;display:flex;flex-direction:column;overflow-y:auto}
.side-header{background:var(--orange);padding:0 16px;height:52px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.side-header-logo{display:flex;flex-direction:column;cursor:pointer}
.side-header-logo .sh-name{font-size:18px;font-weight:900;color:white;letter-spacing:1px;line-height:1.1}
.side-header-logo .sh-sub{font-size:8px;font-weight:700;color:rgba(255,255,255,0.8);letter-spacing:2px;text-transform:uppercase}
.side-header-badge{background:rgba(0,0,0,0.2);color:rgba(255,255,255,0.9);font-size:9px;font-weight:700;padding:3px 6px;border-radius:10px;letter-spacing:1px}
.side-section{padding:8px 14px;border-bottom:1px solid var(--border)}
.side-section:last-child{border-bottom:none}
.side-section-title{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.side-section-title::after{content:'';flex:1;height:1px;background:var(--border)}
.mode-bar{display:flex;gap:6px;padding:6px 16px;background:#fafafa;border-top:1px solid var(--border)}
.mode-btn{padding:5px 12px;border-radius:16px;font-size:11px;font-weight:700;cursor:pointer;border:1.5px solid var(--border);color:var(--muted);transition:all 0.15s;white-space:nowrap}
.mode-btn:hover{border-color:var(--orange);color:var(--orange)}
.mode-btn.active{background:var(--orange);color:white;border-color:var(--orange)}
.cat-items{display:flex;flex-direction:column;gap:3px;margin-bottom:8px}
.cat-item{display:flex;align-items:center;gap:6px;padding:4px 6px;border-radius:3px;font-size:11px;color:var(--text);background:var(--gray);border:1px solid var(--border)}
.cat-item-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cat-item-pages{color:var(--muted);font-size:10px;flex-shrink:0}
.upload-btn{display:flex;align-items:center;justify-content:center;gap:6px;width:100%;padding:8px 10px;border:2px dashed var(--border);border-radius:4px;background:var(--gray);cursor:pointer;font-size:11px;color:var(--muted);font-family:inherit;font-weight:700;transition:all 0.2s}
.upload-btn:hover{border-color:var(--orange);color:var(--orange);background:#fff3ee}
.upload-btn input{display:none}
.upload-progress{display:none;margin-top:6px}
.upload-bar{height:3px;background:var(--gray2);border-radius:2px;overflow:hidden}
.upload-fill{height:100%;background:var(--orange);width:0%;transition:width 0.3s}
.upload-text{font-size:10px;color:var(--muted);margin-top:3px;text-align:center}
.upload-ok{margin-top:5px;font-size:11px;color:#27ae60;font-weight:700;display:none}
.upload-err{margin-top:5px;font-size:11px;color:#c0392b;display:none}
.contact-link{display:flex;align-items:center;gap:8px;color:var(--muted);text-decoration:none;font-size:11px;padding:5px 0;border-bottom:1px solid var(--border);transition:color 0.15s}
.contact-link:last-child{border-bottom:none}
.contact-link:hover{color:var(--orange)}
.c-icon{width:20px;height:20px;display:flex;align-items:center;justify-content:center;border-radius:3px;flex-shrink:0}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0}
.chat-header{height:48px;background:white;border-bottom:2px solid var(--orange);display:flex;align-items:center;padding:0 20px;gap:12px;flex-shrink:0}
.chat-header-logo{display:flex;align-items:center;gap:8px}
.chat-header-title{font-size:16px;font-weight:800;color:var(--orange);letter-spacing:0.5px}
.chat-header-logo .logo-sub{font-size:11px;font-weight:700;color:rgba(255,255,255,0.8);letter-spacing:1px}
.chat-mode-label{font-size:12px;font-weight:700;color:rgba(255,255,255,0.7)}
.chat-mode-value{font-size:12px;font-weight:800;color:white}
.msgs{flex:1;overflow-y:scroll;padding:20px;display:flex;flex-direction:column;gap:14px;min-height:0}
.m{border-radius:4px;font-size:14px;line-height:1.65;word-break:break-word}
.u{background:var(--orange);color:white;align-self:flex-end;border-bottom-right-radius:0;padding:11px 15px;font-weight:600;white-space:pre-wrap;max-width:75%}
.b{background:var(--white);border:1px solid var(--border);align-self:flex-start;border-bottom-left-radius:0;box-shadow:0 1px 3px rgba(0,0,0,0.06);overflow:visible;width:fit-content;max-width:82%}
.msg-body{padding:13px 15px;overflow-wrap:break-word;position:relative}
.copy-btn{position:absolute;top:6px;right:6px;background:none;border:none;cursor:pointer;opacity:0;transition:opacity 0.15s;font-size:14px;padding:2px 5px;border-radius:4px;color:var(--muted)}
.copy-btn:hover{background:var(--gray);color:var(--orange)}
.b:hover .copy-btn{opacity:0.6}
.copy-btn.copied{opacity:1;color:#27ae60}
.edit-btn{position:absolute;top:4px;right:4px;background:none;border:none;cursor:pointer;opacity:0;transition:opacity 0.15s;font-size:13px;padding:2px 5px;border-radius:4px;color:rgba(255,255,255,0.7)}
.edit-btn:hover{color:white}
.u:hover .edit-btn{opacity:0.8}
.u{position:relative}
.edit-area{width:100%;background:white;color:var(--text);border:none;border-radius:4px;padding:8px;font-family:inherit;font-size:14px;resize:vertical;min-height:40px}
.edit-actions{display:flex;gap:6px;margin-top:6px;justify-content:flex-end}
.edit-actions button{padding:4px 12px;border-radius:4px;border:none;cursor:pointer;font-size:12px;font-weight:600;font-family:inherit}
.edit-send{background:white;color:var(--orange)}
.edit-cancel{background:rgba(255,255,255,0.3);color:white}
.msg-meta{display:flex;align-items:center;gap:8px;padding:6px 13px;background:var(--gray);border-top:1px solid var(--border);flex-wrap:wrap;font-size:11px}
.meta-source{color:var(--muted);font-weight:600}
.meta-source span{color:var(--orange)}
.meta-page a{color:#1a6b3a;font-weight:700;cursor:pointer;text-decoration:none;margin-right:10px}
.meta-page a:hover{text-decoration:underline}
.fiabilite-bar{display:flex;align-items:center;gap:5px;margin-left:auto}
.fiabilite-bar small{font-size:10px;color:var(--muted)}
.fbar{width:55px;height:4px;background:var(--gray2);border-radius:3px;overflow:hidden}
.fbar-fill{height:100%;border-radius:3px}
.msg-suggestions{padding:8px 13px 10px;display:flex;flex-wrap:wrap;gap:6px;border-top:1px solid var(--border)}
.sugg-btn{background:white;border:1px solid var(--orange);border-radius:16px;padding:3px 9px;font-size:11px;color:var(--orange);cursor:pointer;font-family:inherit;font-weight:600;transition:all 0.15s}
.sugg-btn:hover{background:var(--orange);color:white}
.welcome-sugg{background:linear-gradient(135deg,#fff3ee,#fff8f5);border:1.5px solid var(--orange);border-radius:8px;padding:12px 16px;cursor:pointer;margin-top:8px;transition:all 0.15s;display:inline-block}
.welcome-sugg:hover{background:var(--orange);color:white}
.welcome-sugg-label{font-size:10px;font-weight:800;color:var(--orange);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.welcome-sugg:hover .welcome-sugg-label{color:rgba(255,255,255,0.8)}
.welcome-sugg-text{font-size:13px;font-weight:600;color:var(--text)}
.welcome-sugg:hover .welcome-sugg-text{color:white}
.bar{padding:10px 16px;background:var(--white);border-top:1px solid var(--border);display:flex;gap:8px;flex-shrink:0;align-items:center}
.img-upload-btn{cursor:pointer;font-size:20px;padding:4px 6px;border-radius:6px;transition:background 0.15s;flex-shrink:0}
.img-upload-btn:hover{background:#f0f0f0}
textarea{flex:1;background:var(--gray);border:1px solid var(--border);border-radius:4px;padding:9px 13px;color:var(--text);font-size:14px;resize:none;font-family:inherit;outline:none;transition:border-color 0.2s}
textarea:focus{border-color:var(--orange);background:white}
textarea::placeholder{color:var(--muted)}
.sbtn{background:var(--orange);border:none;border-radius:4px;color:white;padding:0 24px;font-size:14px;font-weight:800;font-family:inherit;cursor:pointer;flex-shrink:0;letter-spacing:0.5px;transition:all 0.2s}
.sbtn:hover{background:var(--orange2);transform:scale(1.02)}
.sbtn:disabled{background:var(--gray2);color:var(--muted);cursor:not-allowed;transform:none}
#welcomeModal{position:fixed;inset:0;background:rgba(0,0,0,0.4);z-index:200;display:flex;align-items:center;justify-content:center}
.wmodal{background:var(--white);border-radius:8px;padding:28px 32px;max-width:420px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.2)}
.wmodal h2{font-size:17px;font-weight:800;color:var(--orange);margin-bottom:10px}
.wmodal p{font-size:13px;color:var(--text);line-height:1.65;margin-bottom:8px}
.wmodal ul{font-size:13px;color:var(--text);line-height:1.8;padding-left:18px;margin-bottom:14px}
.wmodal .close-btn{background:var(--orange);color:white;border:none;border-radius:4px;padding:10px 22px;font-size:13px;font-weight:700;font-family:inherit;cursor:pointer;width:100%}
.wmodal .close-btn:hover{background:var(--orange2)}
</style>
</head>
<body>
<!-- WELCOME MODAL -->
<div id="welcomeModal" style="display:none">
  <div class="wmodal">
    <h2>Bienvenue sur l'Assistant Catalogue AMDP</h2>
    <p>Explorez intelligemment nos catalogues outillage :</p>
    <ul>
      <li>Recherchez une reference, dimensions ou gamme produit</li>
      <li>Interrogez le catalogue Snap-on et vos documents PDF</li>
      <li>Analysez vos fichiers Excel (tarifs, stocks, references)</li>
      <li>Combinez catalogue et recherche web pour plus de contexte</li>
    </ul>
    <p style="color:var(--muted);font-size:12px">Version TEST — Vos retours sont bienvenus.</p>
    <button class="close-btn" onclick="fermerBienvenue()">Commencer</button>
  </div>
</div>
<!-- LOGIN -->
<div id="loginScreen">
  <div class="lbox">
    <div class="lbox-logo">
      <img src="data:image/png;base64,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" alt="AMDP" style="height:50px"/>
    </div>
    <input type="text" id="lu" placeholder="Identifiant" autocomplete="username"/>
    <input type="password" id="lp" placeholder="Mot de passe" autocomplete="current-password"/>
    <button class="lbtn" id="lbtn" onclick="doLogin()">Connexion</button>
    <span class="lerr" id="lerr"></span>
  </div>
</div>
<!-- APP -->
<div id="appScreen">
  <div class="side">
    <div class="side-header" onclick="resetChat()" title="Retour accueil" style="cursor:pointer">
      <div class="side-header-logo">
        <img src="/logo.png" alt="AMDP" style="height:24px;filter:brightness(0) invert(1)"/>
        <span class="sh-sub">Made for Industry</span>
      </div>
      <span class="side-header-badge">TEST</span>
    </div>
    <div style="padding:6px 14px 2px">
      <div style="font-size:9px;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:1px;text-transform:uppercase">Mode actif</div>
      <div id="chatModeLabel" style="font-size:13px;font-weight:800;color:#1a1a1a;padding:2px 0">📚 Catalogues</div>
    </div>
    <div class="side-section">
      <div class="side-section-title">Catalogues PDF</div>
      <div id="catalogueList" class="cat-items"></div>
      <label class="upload-btn">
        <input type="file" id="pdfInput" accept=".pdf"/>
        ＋ Ajouter un catalogue PDF
      </label>
      <div class="upload-progress" id="pdfProgress">
        <div class="upload-bar"><div class="upload-fill" id="pdfFill"></div></div>
        <div class="upload-text" id="pdfProgressText">Chargement...</div>
      </div>
      <div class="upload-ok" id="pdfOk"></div>
      <div class="upload-err" id="pdfErr"></div>
    </div>
    <div class="side-section">
      <div class="side-section-title">Fichiers Excel</div>
      <div id="excelList" class="cat-items"></div>
      <label class="upload-btn">
        <input type="file" id="xlsInput" accept=".xlsx,.xls,.csv"/>
        ＋ Ajouter un Excel / CSV
      </label>
      <div class="upload-progress" id="xlsProgress">
        <div class="upload-bar"><div class="upload-fill" id="xlsFill"></div></div>
        <div class="upload-text" id="xlsProgressText">Lecture...</div>
      </div>
      <div class="upload-ok" id="xlsOk"></div>
      <div class="upload-err" id="xlsErr"></div>
    </div>
    <div class="side-section" style="margin-top:auto">
      <div class="side-section-title">Fiche Produit AMDP</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <input id="ficheRef" type="text" placeholder="Référence (ex: T72FOD)" style="padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-size:11px;font-family:inherit"/>
        <input id="ficheMarque" type="text" placeholder="Marque (ex: Snap-on)" style="padding:6px 8px;border:1px solid var(--border);border-radius:4px;font-size:11px;font-family:inherit"/>
        <button onclick="genererFiche()" style="padding:7px 12px;background:var(--orange);color:white;border:none;border-radius:4px;font-size:11px;font-weight:700;cursor:pointer;font-family:inherit">📄 Générer la fiche</button>
        <div id="ficheStatus" style="font-size:10px;color:var(--muted);display:none"></div>
        <a id="ficheDownload" style="display:none;padding:6px 10px;background:#2d7d46;color:white;border-radius:4px;font-size:11px;font-weight:700;text-align:center;text-decoration:none;cursor:pointer">⬇ Télécharger la fiche</a>
      </div>
    </div>
    <div class="side-section">
      <div class="side-section-title">AMDP</div>
      <a class="contact-link" href="mailto:info@amdp.fr">
        <div class="c-icon" style="background:#e8501a">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="white"><path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
        </div>Nous contacter
      </a>
      <a class="contact-link" href="https://amdp.shop" target="_blank">
        <div class="c-icon" style="background:#e8501a">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="white"><path d="M11 17h2v-6h-2v6zm1-15C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zM11 9h2V7h-2v2z"/></svg>
        </div>amdp.shop
      </a>
      <a class="contact-link" href="https://www.linkedin.com/company/amdp-sas/" target="_blank">
        <div class="c-icon" style="background:#0A66C2">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="white"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
        </div>Suivez-nous sur LinkedIn
      </a>
    </div>
  </div>
  <div class="main">
    <div class="chat-header">
      <span class="chat-header-title">🔍 Recherche IA</span>
    </div>
    <div class="msgs" id="msgs"></div>
    <div class="mode-bar">
      <div class="mode-btn active" id="mr1" onclick="setMode('catalogue',1)">📚 Catalogues</div>
      <div class="mode-btn" id="mr3" onclick="setMode('web',3)">🌐 Web</div>
      <div class="mode-btn" id="mr2" onclick="setMode('excel',2)">📊 Excel</div>
    </div>
    <div class="bar">
      <label class="img-upload-btn" title="Recherche par image">
        <input type="file" id="imgInput" accept="image/*" style="display:none"/>
        📷
      </label>
      <textarea id="inp" rows="2" placeholder="Référence, caractéristiques, dimensions, tarif..."></textarea>
      <button class="sbtn" id="sbtn">Envoyer</button>
    </div>
    <div id="imgPreview" style="display:none;padding:4px 16px;background:#f9f9f9;border-top:1px solid #eee">
      <img id="imgThumb" style="max-height:60px;border-radius:4px;border:1px solid #ddd"/>
      <button onclick="clearImage()" style="margin-left:8px;border:none;background:none;color:#e8501a;cursor:pointer;font-weight:700">✕</button>
    </div>
  </div>
</div>

<script>
function doLogin(){
  var u=document.getElementById('lu').value.trim();
  var p=document.getElementById('lp').value.trim();
  var err=document.getElementById('lerr'),btn=document.getElementById('lbtn');
  err.style.display='none';
  if(!u||!p){err.textContent='Remplissez tous les champs.';err.style.display='block';return;}
  btn.textContent='...';btn.disabled=true;
  var xhr=new XMLHttpRequest();
  xhr.open('GET','/config',true);
  xhr.onload=function(){
    btn.textContent='Connexion';btn.disabled=false;
    if(xhr.status===200){
      var d=JSON.parse(xhr.responseText);
      if(u.toLowerCase()===d.u&&p===d.p){
        document.getElementById('loginScreen').style.display='none';
        document.getElementById('appScreen').style.display='flex';
        initApp();
        chargerCatalogues();
        document.getElementById('welcomeModal').style.display='flex';
      }else{err.textContent='Identifiant ou mot de passe incorrect.';err.style.display='block';}
    }else{err.textContent='Erreur '+xhr.status+'. Reessayez.';err.style.display='block';}
  };
  xhr.onerror=function(){
    btn.textContent='Connexion';btn.disabled=false;
    err.textContent='Serveur en demarrage, attendez 30s et reessayez.';err.style.display='block';
  };
  xhr.send();
}
document.getElementById('lp').onkeydown=function(e){if(e.key==='Enter')doLogin();};
document.getElementById('lu').onkeydown=function(e){if(e.key==='Enter')document.getElementById('lp').focus();};

var currentMode='catalogue';
var inp,sbtn,msgs;
var lastReference='';
var lastPageNum=null;
var lastQuestion='';
var lastSuggestion='';
var MODE_LABELS={
  'catalogue':'📚 Catalogues',
  'excel':'📊 Fichiers Excel',
  'web':'🌐 Web'
};

function fermerBienvenue(){
  document.getElementById('welcomeModal').style.display='none';
  afficherBienvenue();
}

function afficherBienvenue(){
  var xhr=new XMLHttpRequest();
  xhr.open('GET','/welcome-suggestion',true);
  xhr.onload=function(){
    var sugg='';
    if(xhr.status===200){
      var d=JSON.parse(xhr.responseText);
      sugg=d.suggestion||'';
    }
    ajouterBotMsg(
      'Bonjour ! Je suis votre assistant catalogue AMDP. Posez-moi une question sur un produit, une reference ou une gamme.',
      [],[],null,null,null,false,''
    );
    if(inp)inp.focus();
  };
  xhr.onerror=function(){
    ajouterBotMsg('Bonjour ! Que recherchez-vous ?',[],[],null,null,null,false,'');
    if(inp)inp.focus();
  };
  xhr.send();
}

function resetChat(){
  if(!msgs)return;
  msgs.innerHTML='';
  lastReference='';lastPageNum=null;lastQuestion='';
  afficherBienvenue();
}

function setMode(mode,idx){
  currentMode=mode;
  for(var i=1;i<=3;i++){
    var el=document.getElementById('mr'+i);
    if(el)el.classList.remove('active');
  }
  var active=document.getElementById('mr'+idx);
  if(active)active.classList.add('active');
  var label=document.getElementById('chatModeLabel');
  if(label)label.textContent=MODE_LABELS[mode]||mode;
}

function brandLogo(nom){
  var n=nom.toLowerCase();
  var style='font-size:9px;font-weight:900;padding:2px 5px;border-radius:3px;margin-right:5px;color:white;flex-shrink:0;';
  // Couleurs par marque connue
  var bg='#555';
  if(n.indexOf('snap')!==-1)bg='#e8501a';
  else if(n.indexOf('facom')!==-1)bg='#1a3a8f';
  else if(n.indexOf('totech')!==-1)bg='#2e7d32';
  else if(n.indexOf('knipex')!==-1)bg='#d32f2f';
  else if(n.indexOf('ingersoll')!==-1)bg='#003087';
  else if(n.indexOf('ftm')!==-1)bg='#1a3a8f';
  else if(n.indexOf('makita')!==-1)bg='#00a0c6';
  else if(n.indexOf('bosch')!==-1)bg='#005eb8';
  else if(n.indexOf('stanley')!==-1)bg='#ffcc00';
  // Première lettre du nom
  var letter=nom.trim().charAt(0).toUpperCase()||'C';
  return '<span style="background:'+bg+';'+style+'">'+letter+'</span>';
}

function chargerCatalogues(){
  var xhr=new XMLHttpRequest();
  xhr.open('GET','/catalogues',true);
  xhr.onload=function(){
    if(xhr.status!==200)return;
    var data=JSON.parse(xhr.responseText);
    var pdfList=document.getElementById('catalogueList');
    pdfList.innerHTML='';
    var pdfs=data.pdfs||[];
    for(var i=0;i<pdfs.length;i++){
      var item=document.createElement('div');
      item.className='cat-item';
      var nom=pdfs[i].nom||'Catalogue '+(i+1);
      item.innerHTML=brandLogo(nom)+'<span class="cat-item-name">'+nom+'</span>'+(pdfs[i].pages?'<span class="cat-item-pages">'+pdfs[i].pages+'p.</span>':'');
      pdfList.appendChild(item);
    }
    var xlList=document.getElementById('excelList');
    xlList.innerHTML='';
    var excels=data.excels||[];
    for(var j=0;j<excels.length;j++){
      var item2=document.createElement('div');
      item2.className='cat-item';
      item2.innerHTML='<span style="background:#1d6f42;color:white;font-size:9px;font-weight:900;padding:2px 5px;border-radius:3px;margin-right:5px;flex-shrink:0;">XLS</span><span class="cat-item-name">'+excels[j].nom+'</span>';
      xlList.appendChild(item2);
    }
  };
  xhr.send();
}

function initApp(){
  inp=document.getElementById('inp');
  sbtn=document.getElementById('sbtn');
  msgs=document.getElementById('msgs');
  if(!inp||!sbtn)return;
  inp.onkeydown=function(e){
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();e.stopPropagation();envoyer();return false;}
  };
  sbtn.onclick=function(){envoyer();};
  var pdfIn=document.getElementById('pdfInput');
  var xlsIn=document.getElementById('xlsInput');
  if(pdfIn)pdfIn.onchange=handlePdfUpload;
  if(xlsIn)xlsIn.onchange=handleXlsUpload;
}

function handlePdfUpload(){
  var file=this.files[0];if(!file)return;
  var prog=document.getElementById('pdfProgress'),fill=document.getElementById('pdfFill');
  var progTxt=document.getElementById('pdfProgressText');
  var ok=document.getElementById('pdfOk'),err2=document.getElementById('pdfErr');
  ok.style.display='none';err2.style.display='none';
  prog.style.display='block';fill.style.width='0%';
  var pct=0;
  var interval=setInterval(function(){
    pct+=2;if(pct>90)pct=90;fill.style.width=pct+'%';
    progTxt.textContent=pct<20?'Envoi...':pct<40?'Extraction texte...':pct<70?'Indexation IA...':'Finalisation...';
  },300);
  var fd=new FormData();fd.append('file',file);
  var xhr=new XMLHttpRequest();xhr.open('POST','/upload-pdf',true);
  xhr.onload=function(){
    clearInterval(interval);fill.style.width='100%';
    setTimeout(function(){prog.style.display='none';fill.style.width='0%';},800);
    if(xhr.status===200){
      var d=JSON.parse(xhr.responseText);
      if(d.ok){ok.textContent='✓ '+d.message;ok.style.display='block';chargerCatalogues();}
      else{err2.textContent=d.error||'Erreur';err2.style.display='block';}
    }else{err2.textContent='Erreur serveur.';err2.style.display='block';}
  };
  xhr.onerror=function(){clearInterval(interval);prog.style.display='none';err2.textContent='Erreur connexion.';err2.style.display='block';};
  xhr.send(fd);this.value='';
}

function handleXlsUpload(){
  var file=this.files[0];if(!file)return;
  var prog=document.getElementById('xlsProgress'),fill=document.getElementById('xlsFill');
  var progTxt=document.getElementById('xlsProgressText');
  var ok=document.getElementById('xlsOk'),err2=document.getElementById('xlsErr');
  ok.style.display='none';err2.style.display='none';
  prog.style.display='block';fill.style.width='0%';
  var pct=0;
  var interval=setInterval(function(){
    pct+=5;if(pct>90)pct=90;fill.style.width=pct+'%';
    progTxt.textContent=pct<50?'Lecture...':'Traitement...';
  },150);
  var fd=new FormData();fd.append('file',file);
  var xhr=new XMLHttpRequest();xhr.open('POST','/upload-excel',true);
  xhr.onload=function(){
    clearInterval(interval);fill.style.width='100%';
    setTimeout(function(){prog.style.display='none';fill.style.width='0%';},600);
    if(xhr.status===200){
      var d=JSON.parse(xhr.responseText);
      if(d.ok){ok.textContent='✓ '+d.message;ok.style.display='block';chargerCatalogues();}
      else{err2.textContent=d.error||'Erreur';err2.style.display='block';}
    }else{err2.textContent='Erreur serveur.';err2.style.display='block';}
  };
  xhr.onerror=function(){clearInterval(interval);prog.style.display='none';err2.textContent='Erreur connexion.';err2.style.display='block';};
  xhr.send(fd);this.value='';
}

function fbarColor(s){return s>=80?'#27ae60':s>=60?'#f39c12':'#e74c3c';}
function renderText(txt){
  if(!txt)return'';
  var lines=txt.split('\n'),html='';
  for(var i=0;i<lines.length;i++){
    var l=lines[i];
    l=l.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
    l=l.replace(/^- /,'• ');
    // Convert URLs to clickable links
    l=l.replace(/(https?:\/\/[^\s<]+)/g,'<a href="$1" target="_blank" style="color:var(--orange);font-weight:700;text-decoration:underline;">$1</a>');
    if(!l.trim()){html+='<br>';continue;}
    html+='<p style="margin-bottom:5px">'+l+'</p>';
  }
  return html;
}
function extractReference(txt){
  var m=txt.match(/\b([A-Z]{1,4}[0-9]{2,6}[A-Z0-9]*)\b/);
  return m?m[1]:'';
}
function enrichSuggestions(suggs,texte,pageNum,fromExcel){
  var result=[];
  if(fromExcel){
    var defaults=['Afficher toutes les références','Rechercher un prix'];
    var candidates=(suggs&&suggs.length>0?suggs:defaults);
    // Filter out last suggestion
    candidates=candidates.filter(function(s){return s!==lastSuggestion;});
    if(candidates.length===0)candidates=defaults;
    var final=candidates.slice(0,1);
    if(final[0])lastSuggestion=final[0];
    return final;
  }
  var ref=extractReference(texte||'');
  var candidates=[];
  if(ref&&ref!==lastReference){
    candidates.push("Caractéristiques de "+ref);
  }
  if(lastReference){
    candidates.push("Équivalence de "+lastReference);
  }
  for(var i=0;i<(suggs||[]).length;i++){
    var s=suggs[i];
    if(s.toLowerCase().indexOf('voir la page')!==-1)continue;
    candidates.push(s);
  }
  candidates.push('Explorer la gamme complète','Rechercher sur le web');
  // Filter: no duplicates, no same as last suggestion
  var seen={};
  for(var k=0;k<candidates.length&&result.length<1;k++){
    var c=candidates[k];
    if(seen[c]||c===lastSuggestion)continue;
    seen[c]=true;
    result.push(c);
  }
  if(result[0])lastSuggestion=result[0];
  return result.slice(0,1);
}

function ajouterBotMsg(texte,sources,suggestions,fiabilite,pageNum,imageB64,fromExcel,welcomeSugg,pageNums){
  var d=document.createElement('div');
  d.className='m b';
  var body=document.createElement('div');
  body.className='msg-body';
  body.innerHTML=renderText(texte);
  var cpBtn=document.createElement('button');
  cpBtn.className='copy-btn';cpBtn.textContent='\u{1F4CB}';cpBtn.title='Copier';
  (function(t,b){cpBtn.onclick=function(e){e.stopPropagation();navigator.clipboard.writeText(t).then(function(){b.textContent='Copié !';b.classList.add('copied');setTimeout(function(){b.textContent='\u{1F4CB}';b.classList.remove('copied');},1500);});}})(texte,cpBtn);
  body.appendChild(cpBtn);
  if(welcomeSugg){
    var card=document.createElement('div');
    card.className='welcome-sugg';
    card.innerHTML='<div class="welcome-sugg-label">Suggestion pour commencer</div>'
      +'<div class="welcome-sugg-text">'+welcomeSugg+'</div>';
    (function(q){card.onclick=function(){inp.value=q;envoyer();};})(welcomeSugg);
    body.appendChild(card);
  }
  if(imageB64){
    var img=document.createElement('img');
    img.src='data:image/png;base64,'+imageB64;
    img.style.cssText='display:block;max-width:100%;border-radius:4px;margin-top:10px;cursor:pointer;border:1px solid #ddd;';
    img.onclick=function(){
      var w=window.open('','_blank');
      w.document.write('<html><body style="margin:0;background:#000"><img src="'+img.src+'" style="max-width:100%;display:block;margin:auto"></body></html>');
      w.document.close();
    };
    body.appendChild(img);
  }
  d.appendChild(body);
  if(sources&&sources.length>0){
    var meta=document.createElement('div');
    meta.className='msg-meta';
    var src=document.createElement('div');
    src.className='meta-source';
    src.innerHTML='Source : <span>'+sources.join(' + ')+'</span>';
    meta.appendChild(src);
    // Show download buttons for ALL cited pages — with catalogue name
    var pages=pageNums&&pageNums.length>0?pageNums:(pageNum?[pageNum]:[]);
    if(pages.length>0&&!fromExcel&&sources[0]!=='Recherche web'){
      var pg=document.createElement('div');
      pg.className='meta-page';
      var catName=sources[0]||'Catalogue';
      var safeCat=catName.replace(/'/g,'');
      var pgHtml='';
      for(var pi=0;pi<pages.length;pi++){
        pgHtml+='<a onclick="telechargerPage('+pages[pi]+',\''+safeCat+'\')">'+catName+' ⬇ Page '+pages[pi]+'</a>';
      }
      pg.innerHTML=pgHtml;
      meta.appendChild(pg);
    }
    if(fiabilite!=null){
      var fb=document.createElement('div');fb.className='fiabilite-bar';
      var fbl=document.createElement('small');fbl.textContent='Fiabilité '+fiabilite+'/100';
      var fbar=document.createElement('div');fbar.className='fbar';
      var ffill=document.createElement('div');ffill.className='fbar-fill';
      ffill.style.width=fiabilite+'%';ffill.style.background=fbarColor(fiabilite);
      fbar.appendChild(ffill);fb.appendChild(fbl);fb.appendChild(fbar);
      meta.appendChild(fb);
    }
    d.appendChild(meta);
  }
  var dynSuggs=enrichSuggestions(suggestions,texte,pageNum,fromExcel);
  if(dynSuggs&&dynSuggs.length>0){
    var srow=document.createElement('div');
    srow.className='msg-suggestions';
    for(var i=0;i<dynSuggs.length;i++){
      var sb=document.createElement('button');
      sb.className='sugg-btn';sb.textContent=dynSuggs[i];
      (function(t){sb.onclick=function(){
        var isWeb=['rechercher sur le web','web','internet'].some(function(w){return t.toLowerCase().indexOf(w)!==-1;});
        if(isWeb){
          setMode('web',3);
          // Use lastQuestion (the real last question, not the suggestion text)
          inp.value=lastQuestion||t;
          envoyer();
          return;
        }
        inp.value=t;envoyer();
      };})(dynSuggs[i]);
      srow.appendChild(sb);
    }
    d.appendChild(srow);
  }
  if(pageNum)lastPageNum=pageNum;
  var ref=extractReference(texte);if(ref){lastReference=ref;}else if(fiabilite&&fiabilite<50){lastReference='';}
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
  return d;
}

function telechargerPage(num,cat){
  var url='/download-page/'+num;
  if(cat)url+='?catalogue='+encodeURIComponent(cat);
  var a=document.createElement('a');a.href=url;a.target='_blank';a.rel='noopener';
  document.body.appendChild(a);a.click();setTimeout(function(){document.body.removeChild(a);},100);
}

function genererFiche(){
  var ref=document.getElementById('ficheRef').value.trim();
  var marque=document.getElementById('ficheMarque').value.trim();
  if(!ref){alert('Veuillez saisir une référence.');return;}
  var st=document.getElementById('ficheStatus');
  var dl=document.getElementById('ficheDownload');
  st.style.display='block';st.textContent='Génération en cours...';st.style.color='var(--orange)';
  dl.style.display='none';
  var xhr=new XMLHttpRequest();
  xhr.open('POST','/fiche-produit',true);
  xhr.setRequestHeader('Content-Type','application/json');
  xhr.responseType='blob';
  xhr.onload=function(){
    if(xhr.status===200){
      var blob=xhr.response;
      var url=URL.createObjectURL(blob);
      dl.href=url;dl.download='Fiche_'+ref+'.pdf';dl.style.display='block';
      st.textContent='Fiche générée !';st.style.color='#2d7d46';
    }else{st.textContent='Erreur lors de la génération.';st.style.color='red';}
  };
  xhr.onerror=function(){st.textContent='Erreur de connexion.';st.style.color='red';};
  xhr.send(JSON.stringify({reference:ref,marque:marque}));
}

// Template upload
var templateInput=document.getElementById('templateInput');
if(templateInput){
  templateInput.addEventListener('change',function(e){
    var file=e.target.files[0];
    if(!file)return;
    var tst=document.getElementById('templateStatus');
    tst.style.display='block';tst.textContent='Envoi...';tst.style.color='var(--orange)';
    var fd=new FormData();fd.append('file',file);
    var xhr=new XMLHttpRequest();
    xhr.open('POST','/upload-template',true);
    xhr.onload=function(){
      var r=JSON.parse(xhr.responseText);
      if(r.ok){tst.textContent='✓ Modèle enregistré';tst.style.color='#2d7d46';}
      else{tst.textContent='Erreur: '+r.error;tst.style.color='red';}
    };
    xhr.send(fd);
    templateInput.value='';
  });
}

var pendingImage=null;
var imgInput=document.getElementById('imgInput');
var imgPreview=document.getElementById('imgPreview');
var imgThumb=document.getElementById('imgThumb');

if(imgInput){
  imgInput.addEventListener('change',function(e){
    var file=e.target.files[0];
    if(!file)return;
    var reader=new FileReader();
    reader.onload=function(ev){
      pendingImage=ev.target.result; // data:image/...;base64,...
      imgThumb.src=pendingImage;
      imgPreview.style.display='flex';
      imgPreview.style.alignItems='center';
      if(!inp.value.trim())inp.value='Identifie cet outil et trouve sa référence';
      inp.focus();
    };
    reader.readAsDataURL(file);
    imgInput.value='';
  });
}
function clearImage(){pendingImage=null;imgPreview.style.display='none';}

function editMessage(um,origText){
  var allMsgs=Array.from(msgs.children);
  var idx=allMsgs.indexOf(um);
  um.innerHTML='';um.style.maxWidth='85%';
  var ta=document.createElement('textarea');ta.className='edit-area';ta.value=origText;
  var acts=document.createElement('div');acts.className='edit-actions';
  var sendBtn=document.createElement('button');sendBtn.className='edit-send';sendBtn.textContent='Envoyer';
  var cancelBtn=document.createElement('button');cancelBtn.className='edit-cancel';cancelBtn.textContent='Annuler';
  acts.appendChild(cancelBtn);acts.appendChild(sendBtn);
  um.appendChild(ta);um.appendChild(acts);
  ta.focus();ta.setSelectionRange(ta.value.length,ta.value.length);
  cancelBtn.onclick=function(){um.innerHTML='';um.textContent=origText;addEditBtn(um,origText);um.style.maxWidth='75%';};
  sendBtn.onclick=function(){
    var nq=ta.value.trim();if(!nq)return;
    // Remove all messages after this one
    while(msgs.lastChild&&msgs.lastChild!==um)msgs.removeChild(msgs.lastChild);
    um.innerHTML='';um.textContent=nq;addEditBtn(um,nq);um.style.maxWidth='75%';
    inp.value=nq;envoyer();
  };
}
function addEditBtn(um,txt){
  var eb=document.createElement('button');eb.className='edit-btn';eb.textContent='\u270F\uFE0F';eb.title='Modifier';
  eb.onclick=function(e){e.stopPropagation();editMessage(um,txt);};
  um.appendChild(eb);
}

function envoyer(){
  if(!inp||!sbtn)return;
  var q=inp.value.trim();
  if(!q&&!pendingImage||sbtn.disabled)return;
  if(!q&&pendingImage)q='Identifie cet outil et trouve sa référence';
  lastQuestion=q;
  var um=document.createElement('div');um.className='m u';
  if(pendingImage){
    um.innerHTML='<div>'+q+'</div><img src="'+pendingImage+'" style="max-height:80px;border-radius:4px;margin-top:6px"/>';
  }else{um.textContent=q;addEditBtn(um,q);}
  msgs.appendChild(um);msgs.scrollTop=msgs.scrollHeight;
  inp.value='';sbtn.disabled=true;
  var attente=document.createElement('div');attente.className='m b';
  attente.innerHTML='<div class="msg-body" style="color:#aaa;font-style:italic">Recherche en cours...<br><a class="cancel-link" style="font-size:11px;color:#bbb;cursor:pointer;text-decoration:none;transition:color 0.15s">Annuler</a></div>';
  msgs.appendChild(attente);msgs.scrollTop=msgs.scrollHeight;
  var activeXhr=null;
  attente.querySelector('.cancel-link').onclick=function(){if(activeXhr){activeXhr.abort();attente.remove();sbtn.disabled=false;inp.focus();}};

  if(pendingImage){
    // Send image via /question-image endpoint
    var xhr=new XMLHttpRequest();activeXhr=xhr;
    xhr.open('POST','/question-image',true);
    xhr.setRequestHeader('Content-Type','application/json');
    xhr.onload=function(){
      attente.remove();
      if(xhr.status===200){
        var data=JSON.parse(xhr.responseText);
        // After image analysis, switch to web mode for follow-up
        if(data.propose_web){setMode('web',3);}
        else if(data.mode_used){var mm={'catalogue':1,'excel':2,'web':3};var idx=mm[data.mode_used];if(idx)setMode(data.mode_used,idx);}
        ajouterBotMsg(data.reponse,data.sources,data.suggestions,data.fiabilite,data.page_num,data.image,data.from_excel||false,'',data.page_nums||[]);
      }else{ajouterBotMsg('Erreur serveur. Réessayez.',[],[],null,null,null,false,'');}
      sbtn.disabled=false;inp.focus();
    };
    xhr.onerror=function(){attente.remove();ajouterBotMsg('Erreur de connexion.',[],[],null,null,null,false,'');sbtn.disabled=false;};
    xhr.send(JSON.stringify({question:q,mode:currentMode,image:pendingImage}));
    clearImage();
  }else{
    var xhr=new XMLHttpRequest();activeXhr=xhr;
    xhr.open('POST','/question',true);
    xhr.setRequestHeader('Content-Type','application/json');
    xhr.onload=function(){
      attente.remove();
      if(xhr.status===200){
        var data=JSON.parse(xhr.responseText);
        if(data.mode_used && data.mode_used!==currentMode && data.mode_used==='web'){var mm={'catalogue':1,'excel':2,'web':3};setMode(data.mode_used,mm[data.mode_used]);}
        ajouterBotMsg(data.reponse,data.sources,data.suggestions,data.fiabilite,data.page_num,data.image,data.from_excel||false,'',data.page_nums||[]);
      }else{ajouterBotMsg('Erreur serveur. Réessayez.',[],[],null,null,null,false,'');}
      sbtn.disabled=false;inp.focus();
    };
    xhr.onerror=function(){attente.remove();ajouterBotMsg('Erreur de connexion.',[],[],null,null,null,false,'');sbtn.disabled=false;};
    xhr.send(JSON.stringify({question:q,mode:currentMode}));
  }
}
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════
ids_global = None

import datetime
print(f"===== Application Startup at {datetime.datetime.now()} =====", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ids_global
    print("[BOOT] Lifespan started...", flush=True)
    try:
        ids_global = charger_fichiers()
        print("[BOOT] Demarrage OK !", flush=True)
    except Exception as e:
        import traceback
        print(f"[BOOT] ERREUR: {e}", flush=True)
        traceback.print_exc()
        raise
    yield
    print("[BOOT] Shutdown.", flush=True)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def accueil():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/logo.png")
def serve_logo():
    logo_path = os.path.join(BASE_DIR, "logo_amdp.png")
    if os.path.exists(logo_path):
        return StreamingResponse(open(logo_path, "rb"), media_type="image/png")
    # Also try Logo_AMDP_2022.png
    alt_path = os.path.join(BASE_DIR, "Logo_AMDP_2022.png")
    if os.path.exists(alt_path):
        return StreamingResponse(open(alt_path, "rb"), media_type="image/png")
    # SVG fallback
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 60">
    <text x="0" y="45" font-family="Arial" font-weight="900" font-size="48" fill="#1B2A4A">AM</text>
    <text x="95" y="45" font-family="Arial" font-weight="900" font-size="48" fill="#E8301E">D</text>
    <text x="135" y="45" font-family="Arial" font-weight="900" font-size="48" fill="#1B2A4A">P</text>
    </svg>"""
    return StreamingResponse(io.BytesIO(svg.encode()), media_type="image/svg+xml")

@app.get("/config")
def config():
    return {"u": LOGIN_USER.strip(), "p": LOGIN_PASS.strip()}

@app.get("/welcome-suggestion")
def welcome_suggestion():
    sugg = generer_question_bienvenue()
    return {"suggestion": sugg}

@app.get("/suggestions")
def suggestions():
    sug = generer_suggestions()
    return [{"label": s[0], "text": s[1]} for s in sug]

@app.get("/catalogues")
def get_catalogues():
    pdfs = [{"nom": CATALOGUE_NOM_PRINCIPAL,
             "pages": len(set(p["page"] for p in vector_db.ids_main)) if vector_db.ids_main else 0}]
    for nom, data in pdfs_supplementaires.items():
        pdfs.append({"nom": data["nom_affichage"], "pages": data["texte"].count("[Page ")})
    excels = []
    # From SQLite tables (primary source)
    seen = set()
    for fn, info in excel_db.tables.items():
        excels.append({"nom": info["nom_affichage"], "rows": info["rows"]})
        seen.add(fn)
    # Legacy fallback
    for nom, data in excels_supplementaires.items():
        if nom not in seen:
            excels.append({"nom": data["nom_affichage"]})
    return {"pdfs": pdfs, "excels": excels}

@app.get("/download-page/{page_num}")
def download_page(page_num: int, catalogue: str = ""):
    """Download a page from the main catalogue or a supplementary one."""
    try:
        # Try main catalogue first
        pdf_path = PDF_LOCAL
        cat_name = CATALOGUE_NOM_PRINCIPAL

        # If a specific catalogue is requested, find it
        if catalogue:
            cat_clean = catalogue.lower().replace(".pdf", "").strip()
            for fname, data in pdfs_supplementaires.items():
                nom_aff = data["nom_affichage"].lower()
                fname_clean = fname.lower().replace(".pdf", "")
                if cat_clean in nom_aff or cat_clean in fname_clean or nom_aff in cat_clean:
                    supp_path = os.path.join(BASE_DIR, fname)
                    if os.path.exists(supp_path):
                        pdf_path = supp_path
                        cat_name = data["nom_affichage"]
                        print(f"[DOWNLOAD] Matched catalogue: {cat_name} → {supp_path}")
                        break

        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            # Try supplementary PDFs if page not found in main
            for fname, data in pdfs_supplementaires.items():
                supp_path = os.path.join(BASE_DIR, fname)
                if os.path.exists(supp_path):
                    try:
                        sdoc = fitz.open(supp_path)
                        if page_num >= 1 and page_num <= len(sdoc):
                            new_doc = fitz.open()
                            new_doc.insert_pdf(sdoc, from_page=page_num-1, to_page=page_num-1)
                            pdf_bytes = new_doc.tobytes()
                            sdoc.close(); new_doc.close()
                            safe_name = data["nom_affichage"].replace(" ", "_")
                            return StreamingResponse(io.BytesIO(pdf_bytes),
                                media_type="application/pdf",
                                headers={"Content-Disposition":
                                    f"attachment; filename={safe_name}_page_{page_num}.pdf"})
                        sdoc.close()
                    except Exception:
                        pass
            return {"error": "Page invalide"}

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
        pdf_bytes = new_doc.tobytes()
        doc.close(); new_doc.close()
        safe_name = cat_name.replace(" ", "_")
        return StreamingResponse(io.BytesIO(pdf_bytes),
                                  media_type="application/pdf",
                                  headers={"Content-Disposition":
                                           f"attachment; filename={safe_name}_page_{page_num}.pdf"})
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdfs_supplementaires
    try:
        if not file.filename.lower().endswith(".pdf"):
            return {"ok": False, "error": "Le fichier doit être un PDF."}
        contenu = await file.read()

        # Extract text (for display/fallback)
        texte = extraire_texte_pdf(contenu)
        if not texte.strip():
            return {"ok": False, "error": "Impossible d'extraire le texte de ce PDF."}

        nom_affichage = file.filename.replace(".pdf","").replace("_"," ").replace("-"," ").strip()
        pdfs_supplementaires[file.filename] = {"texte": texte, "nom_affichage": nom_affichage}

        # Save PDF file to disk for page download
        pdf_save_path = os.path.join(BASE_DIR, file.filename)
        with open(pdf_save_path, "wb") as f:
            f.write(contenu)

        # Smart chunking + vector indexing
        chunks = chunker_pdf_intelligent(contenu, chunk_size=800, overlap=200)
        if chunks:
            vector_db.index_supplementary_pdf(file.filename, chunks)

        nb_pages = texte.count("[Page ")
        nb_chunks = len(chunks) if chunks else 0
        return {"ok": True,
                "message": f"{nom_affichage} chargé ({nb_pages} pages, {nb_chunks} extraits indexés)",
                "pages": nb_pages, "nom_affichage": nom_affichage}
    except Exception as e:
        import traceback; print("PDF UPLOAD ERROR:", traceback.format_exc())
        return {"ok": False, "error": str(e)}

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    global excels_supplementaires
    try:
        fname = file.filename.lower()
        if not (fname.endswith(".xlsx") or fname.endswith(".xls") or fname.endswith(".csv")):
            return {"ok": False, "error": "Format non supporté. Utilisez .xlsx .xls ou .csv"}
        contenu = await file.read()
        nom_affichage = file.filename.rsplit(".",1)[0].replace("_"," ").replace("-"," ").strip()
        lignes_txt = []
        if fname.endswith(".csv"):
            texte = contenu.decode("utf-8", errors="ignore")
            lignes_txt = texte.splitlines()
        else:
            if _opxl_global is None:
                return {"ok": False, "error": "openpyxl non disponible sur ce serveur"}
            wb = _opxl_global.load_workbook(io.BytesIO(contenu), read_only=True, data_only=True)
            for ws in wb.worksheets:
                lignes_txt.append("[Feuille: " + ws.title + "]")
                for row in ws.iter_rows(values_only=True):
                    row_vals = [str(c) if c is not None else "" for c in row]
                    if any(v.strip() for v in row_vals):
                        lignes_txt.append(" | ".join(row_vals))
            wb.close()
        texte_final = "\n".join(lignes_txt)
        excels_supplementaires[file.filename] = {"texte": texte_final, "nom_affichage": nom_affichage}

        # Also import into SQLite for Text-to-SQL
        try:
            excel_db.import_excel(file.filename, lignes_txt, nom_affichage)
        except Exception as e:
            print(f"SQLite import warning: {e}")

        return {"ok": True,
                "message": nom_affichage + " chargé (" + str(len(lignes_txt)) + " lignes)",
                "lignes": len(lignes_txt), "nom_affichage": nom_affichage}
    except Exception as e:
        import traceback; print("EXCEL ERROR:", traceback.format_exc())
        return {"ok": False, "error": "Erreur lecture fichier: " + str(e)[:150]}

class Question(BaseModel):
    question: str
    mode: str = "catalogue"

@app.post("/question")
def question(body: Question):
    try:
        result = poser_question(body.question, mode=body.mode)
    except Exception as e:
        import traceback
        print("ERROR in poser_question:", traceback.format_exc())
        return {"question": body.question, "reponse": "Erreur interne: " + str(e)[:200],
                "image": None, "sources": [], "fiabilite": 0, "suggestions": [],
                "page_num": None, "from_excel": False, "propose_web": False}
    return {
        "question":   body.question,
        "reponse":    result["texte"],
        "image":      result.get("image"),
        "sources":    result.get("sources", []),
        "fiabilite":  result.get("fiabilite", 70),
        "suggestions":result.get("suggestions", []),
        "page_num":   result.get("page_num"),
        "page_nums":  result.get("page_nums", []),
        "from_excel": result.get("from_excel", False),
        "propose_web":result.get("propose_web", False),
        "mode_used":  result.get("mode_used", "catalogue")
    }

class QuestionImage(BaseModel):
    question: str
    mode: str = "catalogue"
    image: str = ""  # base64 data URL

@app.post("/question-image")
def question_image(body: QuestionImage):
    """Handle image-based questions using GPT-4o vision."""
    global resume_conversation, derniere_question, dernier_contexte_envoye, derniere_propose_web
    global historique_conversation
    try:
        # Extract base64 from data URL
        img_data = body.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        # Step 1: Use GPT-4o vision to identify the tool/product in the image
        vision_msgs = [
            {"role": "system", "content":
                "Tu es expert en outillage professionnel. Analyse cette image et identifie l'outil ou produit. "
                "Donne: nom précis, type, marque si visible, référence si visible, caractéristiques observables. "
                "Si tu vois une marque, essaie de deviner la gamme ou série de produits. "
                "Réponds en français, de façon concise."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                {"type": "text", "text": body.question or "Identifie cet outil et trouve sa référence"}
            ]}
        ]
        r = client.chat.completions.create(model="gpt-4o", max_tokens=500, messages=vision_msgs)
        vision_answer = r.choices[0].message.content

        # Step 2: Search in catalogue with identified product
        # Extract potential reference from vision answer
        ref_match = re.search(r'\b([A-Z]{1,4}[0-9]{2,6}[A-Z0-9]*)\b', vision_answer)
        search_query = ref_match.group(1) if ref_match else vision_answer[:100]

        # Search catalogue
        result = poser_question(
            f"{body.question}. Description de l'image: {vision_answer[:200]}",
            mode=body.mode
        )

        # Combine: if catalogue found something, enrich. Otherwise, auto web search.
        catalogue_answer = result.get("texte", "")
        if "AUCUN_RESULTAT" in catalogue_answer or "non trouvé" in catalogue_answer.lower() or result.get("fiabilite", 0) < 40:
            # Catalogue didn't find — automatically search the web with vision description
            web_answer = ""
            try:
                web_query = f"{vision_answer[:150]} référence produit"
                web_answer = recherche_perplexity(web_query)
                if web_answer:
                    web_answer = re.sub(r"\[\d+\]", "", web_answer).strip()
            except Exception:
                pass

            final_answer = f"D'après l'image :\n{vision_answer}"
            if web_answer:
                final_answer += f"\n\nRésultats de la recherche web :\n{web_answer}"

            # Save context for follow-up
            resume_conversation = f"L'utilisateur a envoyé une image. Identification: {vision_answer[:200]}"
            derniere_question = body.question

            return {
                "question": body.question, "reponse": final_answer,
                "image": None, "sources": ["Analyse d'image (GPT-4o)" + (" + Recherche web" if web_answer else "")],
                "fiabilite": 80 if web_answer else 70,
                "suggestions": ["Rechercher sur le web"] if not web_answer else [],
                "page_num": None, "page_nums": [],
                "from_excel": False, "propose_web": not bool(web_answer),
                "mode_used": "web" if web_answer else body.mode
            }
        else:
            # Catalogue found something — combine
            final_answer = f"D'après l'image : {vision_answer[:150]}\n\n{catalogue_answer}"
            result["texte"] = final_answer
            return {
                "question": body.question, "reponse": final_answer,
                "image": result.get("image"),
                "sources": result.get("sources", ["Analyse d'image + Catalogue"]),
                "fiabilite": result.get("fiabilite", 80),
                "suggestions": result.get("suggestions", []),
                "page_num": result.get("page_num"),
                "page_nums": result.get("page_nums", []),
                "from_excel": False, "propose_web": False,
                "mode_used": result.get("mode_used", body.mode)
            }
    except Exception as e:
        import traceback
        print("ERROR in question_image:", traceback.format_exc())
        return {"question": body.question, "reponse": "Erreur analyse image: " + str(e)[:200],
                "image": None, "sources": [], "fiabilite": 0, "suggestions": [],
                "page_num": None, "page_nums": [],
                "from_excel": False, "propose_web": False, "mode_used": body.mode}

@app.post("/upload-template")
async def upload_template(file: UploadFile = File(...)):
    """Upload a fiche produit template PDF."""
    try:
        if not file.filename.lower().endswith(".pdf"):
            return {"ok": False, "error": "Le fichier doit être un PDF."}
        contenu = await file.read()
        with open(FICHE_TEMPLATE, "wb") as f:
            f.write(contenu)
        print(f"[TEMPLATE] Modèle fiche produit sauvegardé: {FICHE_TEMPLATE} ({len(contenu)} bytes)")
        return {"ok": True, "message": f"Template '{file.filename}' enregistré avec succès."}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class FicheProduitRequest(BaseModel):
    reference: str
    marque: str = ""

@app.post("/fiche-produit")
def generer_fiche_produit(body: FicheProduitRequest):
    """Generate a product data sheet PDF for a given reference and brand."""
    try:
        ref = body.reference.strip()
        marque = body.marque.strip() or "Non spécifiée"

        # Step 1: Search in catalogue for product info
        catalogue_info = ""
        try:
            result = poser_question(
                f"Donne toutes les caractéristiques techniques du produit {ref} de la marque {marque}. "
                f"Inclus: dimensions, poids, matériaux, entraînement, capacité, normes.",
                mode="catalogue"
            )
            if result.get("fiabilite", 0) > 30:
                catalogue_info = result.get("texte", "")
        except Exception:
            pass

        # Step 2: Search web for complementary info
        web_info = ""
        try:
            web_info = recherche_perplexity(
                f"{marque} {ref} fiche technique caractéristiques prix"
            )
            if web_info:
                web_info = re.sub(r"\[\d+\]", "", web_info).strip()
        except Exception:
            pass

        # Step 3: Use GPT to structure the product sheet as JSON
        combined_context = f"INFO CATALOGUE:\n{catalogue_info}\n\nINFO WEB:\n{web_info}"
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1500,
            messages=[
                {"role": "system", "content":
                    "Tu es expert en fiches techniques pour AMDP (distributeur d'outillage professionnel). "
                    "Génère une fiche produit au format JSON STRICT. "
                    "Réponds UNIQUEMENT avec un objet JSON valide, RIEN d'autre. "
                    "PAS de texte avant ou après le JSON. PAS de ```json```. "
                    "PAS de markdown (###, **, -, •, \"\"). Texte brut dans les valeurs. "
                    "Format EXACT:\n"
                    '{"nom_produit":"Clé à cliquet 1/4 Dual 80 FOD",'
                    '"libelle":"Cliquet 1/4 Snap-on",'
                    '"reference":"T72FOD",'
                    '"marque":"Snap-on",'
                    '"description_courte":"Clé à cliquet compacte avec technologie Dual 80.",'
                    '"description_longue":"La T72FOD est une clé à cliquet 1/4 avec 72 dents...",'
                    '"caracteristiques":[{"cle":"Type","valeur":"Clé à cliquet"},{"cle":"Entraînement","valeur":"1/4 pouce"}],'
                    '"application":"Aéronautique, maintenance industrielle"}'},
                {"role": "user", "content":
                    f"Produit: {ref} de {marque}\n\n{combined_context}\n\n"
                    f"Génère le JSON de la fiche technique:"}
            ]
        )
        raw_json = r.choices[0].message.content.strip()
        # Clean potential markdown wrapping
        raw_json = re.sub(r'^```json\s*', '', raw_json)
        raw_json = re.sub(r'\s*```$', '', raw_json)
        raw_json = raw_json.strip()
        
        # Parse JSON
        try:
            import json
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON from response
            json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
        
        # Extract and clean all fields
        nom = clean_markdown(data.get("nom_produit", f"{marque} {ref}"))
        libelle = clean_markdown(data.get("libelle", nom))
        ref_clean = clean_markdown(data.get("reference", ref))
        desc_courte = clean_markdown(data.get("description_courte", f"Produit {ref} de {marque}"))
        desc_longue = clean_markdown(data.get("description_longue", 
            catalogue_info[:300] if catalogue_info else (web_info[:300] if web_info else "")))
        application = clean_markdown(data.get("application", "Outillage professionnel"))
        
        # Parse characteristics
        carac_lines = []
        raw_caracs = data.get("caracteristiques", [])
        if isinstance(raw_caracs, list):
            for c in raw_caracs:
                if isinstance(c, dict):
                    k = clean_markdown(str(c.get("cle", "")))
                    v = clean_markdown(str(c.get("valeur", "")))
                    if k and v:
                        carac_lines.append(f"{k}: {v}")
                elif isinstance(c, str):
                    carac_lines.append(clean_markdown(c))
        
        # Add APPLICATION to characteristics if available
        if application and application != "Outillage professionnel":
            carac_lines.append(f"Application: {application}")
        
        # Use LIBELLE as the main title if meaningful, fallback to nom
        titre_fiche = libelle if libelle and len(libelle) > 3 else nom
        # Capitalize first letter
        titre_fiche = titre_fiche[0].upper() + titre_fiche[1:] if titre_fiche else f"{marque} {ref}"

        print(f"[FICHE] Titre: {titre_fiche} | Ref: {ref_clean} | Carac: {len(carac_lines)}")

        # Step 5: Generate PDF using the AMDP template PDF as base
        import io as io_mod

        template_path = os.path.join(BASE_DIR, "fiche_template.pdf")

        # ── Brand logos (static URLs for known brands) ──
        BRAND_LOGOS = {
            "snap-on": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Snap-on_logo.svg/200px-Snap-on_logo.svg.png",
            "bahco": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Bahco_logo.svg/200px-Bahco_logo.svg.png",
            "facom": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Facom_logo.svg/200px-Facom_logo.svg.png",
            "beta": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Beta_Utensili_logo.svg/200px-Beta_Utensili_logo.svg.png",
            "knipex": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Knipex-Logo.svg/200px-Knipex-Logo.svg.png",
            "wera": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Wera_Logo.svg/200px-Wera_Logo.svg.png",
            "stahlwille": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Stahlwille_logo.svg/200px-Stahlwille_logo.svg.png",
        }

        # ── Fetch product image from web ──
        product_image_data = None
        brand_logo_data = None
        import urllib.request

        # Try to get brand logo
        marque_key = marque.lower().replace(" ", "-")
        logo_url = BRAND_LOGOS.get(marque_key)
        if logo_url:
            try:
                req = urllib.request.Request(logo_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    brand_logo_data = resp.read()
                if len(brand_logo_data) < 500:
                    brand_logo_data = None
                else:
                    print(f"[FICHE] Logo marque téléchargé: {len(brand_logo_data)} bytes")
            except Exception as e:
                print(f"[FICHE] Logo marque non disponible: {e}")

        # Try to get product image from Perplexity
        try:
            img_resp = client_perplexity.chat.completions.create(
                model="sonar", max_tokens=300,
                messages=[{"role": "user", "content":
                    f"Find ONE direct image URL (ending in .jpg, .png or .webp) for the product {marque} {ref}. "
                    f"Reply ONLY with the URL, nothing else. If no image found, reply NONE."}]
            )
            img_url = img_resp.choices[0].message.content.strip()
            url_match = re.search(r'https?://\S+\.(?:jpg|jpeg|png|webp)', img_url, re.IGNORECASE)
            if url_match:
                img_url = url_match.group(0).rstrip('.,;)')
            if img_url and img_url.startswith("http") and "NONE" not in img_url.upper():
                req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    product_image_data = resp.read()
                if len(product_image_data) < 1000:
                    product_image_data = None
                else:
                    print(f"[FICHE] Image produit web: {len(product_image_data)} bytes")
        except Exception as e:
            print(f"[FICHE] Pas d'image web: {e}")

        # Fallback: extract the largest individual image from catalogue page
        if not product_image_data:
            try:
                page_num = None
                if result.get("page_num"):
                    page_num = result["page_num"]
                elif derniers_chunks_trouves:
                    page_num = derniers_chunks_trouves[0].get("page")
                if page_num:
                    doc_cat = fitz.open(PDF_LOCAL)
                    cat_page = doc_cat[page_num - 1]
                    images = cat_page.get_images(full=True)
                    best_img = None
                    best_area = 0
                    for img_info in images:
                        xref = img_info[0]
                        try:
                            pix = fitz.Pixmap(doc_cat, xref)
                            w, h = pix.width, pix.height
                            if w > 100 and h > 100 and w * h > best_area:
                                # Convert CMYK/other to RGB if needed
                                if pix.n - pix.alpha > 3:
                                    pix = fitz.Pixmap(fitz.csRGB, pix)
                                best_area = w * h
                                best_img = pix.tobytes("png")
                        except Exception:
                            continue
                    doc_cat.close()
                    if best_img:
                        product_image_data = best_img
                        print(f"[FICHE] Image produit extraite page {page_num} ({best_area}px)")
                    else:
                        print(f"[FICHE] Pas d'image assez grande sur page {page_num}")
            except Exception as e:
                print(f"[FICHE] Pas d'image catalogue: {e}")

        # ── Generate PDF ──
        # Template layout (from inspection):
        #   Header: y=0 to ~95 (logo AMDP banner, image xref=38)
        #   Content: y=100 to ~805
        #     Libellé: y=152, size 30, FuturaLT-Heavy (red)
        #     Product image placeholder: y=203-388, x=34-219 (image xref=44)
        #     Description: y=234, size 18, right of image
        #     Caractéristiques title: y=521, size 22
        #     Info 1/2/3: y=555/579/604, size 18
        #   Footer: y=810 to ~840 (AMDP address, image xref=40)

        blue = (0.129, 0.169, 0.325)
        red = (0.843, 0.165, 0.176)
        black = (0, 0, 0)
        gray = (0.35, 0.35, 0.35)
        white = (1, 1, 1)
        light_gray = (0.94, 0.94, 0.94)

        use_template = False
        try:
            if os.path.exists(template_path):
                doc = fitz.open(template_path)
                page = doc[0]

                # Redact ONLY the content area (preserve header images + footer)
                # Two separate redactions to keep header logo (y<95) and footer (y>805)
                content_rect = fitz.Rect(0, 95, 596, 805)
                annot = page.add_redact_annot(content_rect)
                annot.set_colors(fill=(1, 1, 1))
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)

                # ═══ TITRE (centered, red bold) — matches template y=140 ═══
                y = 120
                # Reference line (small, above title)
                ref_line = f"Réf: {ref_clean}  |  Marque: {marque}"
                tw_ref = fitz.get_text_length(ref_line, fontsize=10, fontname="helv")
                page.insert_text(fitz.Point((595 - tw_ref) / 2, y), ref_line, fontsize=10, fontname="helv", color=gray)
                y += 20

                # Title (large, red, centered)
                title_rect = fitz.Rect(30, y, 565, y + 60)
                page.insert_textbox(title_rect, titre_fiche, fontsize=24, fontname="hebo", color=red, align=1)
                y += 55

                # Separator line
                page.draw_line(fitz.Point(30, y), fitz.Point(565, y), color=red, width=1.5)
                y += 15

                # ═══ IMAGE PRODUIT (left) + DESCRIPTION (right) ═══
                img_y = y
                has_image = False

                if product_image_data:
                    try:
                        # Preserve aspect ratio within 180x180 box
                        img_rect = fitz.Rect(35, img_y, 220, img_y + 180)
                        page.insert_image(img_rect, stream=product_image_data, keep_proportion=True)
                        has_image = True
                    except Exception as e:
                        print(f"[FICHE] Erreur image produit: {e}")

                # Brand logo (below or beside product image)
                if brand_logo_data:
                    try:
                        logo_y = img_y + (185 if has_image else 0)
                        logo_rect = fitz.Rect(35, logo_y, 150, logo_y + 40)
                        page.insert_image(logo_rect, stream=brand_logo_data, keep_proportion=True)
                    except Exception as e:
                        print(f"[FICHE] Erreur logo marque: {e}")

                # Description text — right of image or full width
                if has_image:
                    desc_rect = fitz.Rect(235, img_y, 565, img_y + 180)
                else:
                    desc_rect = fitz.Rect(35, img_y, 565, img_y + 120)

                presentation_text = desc_courte if desc_courte else desc_longue[:500]
                if desc_longue and desc_courte and desc_longue != desc_courte:
                    presentation_text = desc_courte + "\n\n" + desc_longue[:300]
                page.insert_textbox(desc_rect, presentation_text[:600], fontsize=10, fontname="helv", color=black, align=0)

                # ═══ CARACTÉRISTIQUES ═══
                carac_y = img_y + (230 if has_image else 140)

                # Section title
                page.insert_text(fitz.Point(30, carac_y), "Caractéristiques techniques", fontsize=14, fontname="hebo", color=red)
                carac_y += 5
                page.draw_line(fitz.Point(30, carac_y), fitz.Point(350, carac_y), color=red, width=1)
                carac_y += 15

                # Characteristics as a proper table with alternating rows
                if carac_lines:
                    for i, carac in enumerate(carac_lines[:15]):
                        if carac_y > 780:
                            break
                        # Alternating background
                        bg = light_gray if i % 2 == 0 else white
                        row_rect = fitz.Rect(25, carac_y - 2, 570, carac_y + 16)
                        page.draw_rect(row_rect, color=bg, fill=bg)

                        # Split on first ":" for key:value display
                        if ":" in carac:
                            key, val = carac.split(":", 1)
                            page.insert_text(fitz.Point(30, carac_y + 11), key.strip()[:40], fontsize=10, fontname="hebo", color=black)
                            page.insert_text(fitz.Point(230, carac_y + 11), val.strip()[:60], fontsize=10, fontname="helv", color=gray)
                        else:
                            page.insert_text(fitz.Point(30, carac_y + 11), carac.strip()[:80], fontsize=10, fontname="helv", color=black)

                        carac_y += 19
                else:
                    # No characteristics — show application text
                    if application:
                        page.insert_textbox(
                            fitz.Rect(35, carac_y, 565, carac_y + 60),
                            application[:300], fontsize=10, fontname="helv", color=black, align=0
                        )

                pdf_bytes = doc.tobytes()
                doc.close()
                use_template = True
            else:
                raise Exception("No template file")
        except Exception as e:
            print(f"[FICHE] Erreur template: {e}, fallback")
            use_template = False

        if not use_template:
            # === FALLBACK: Generate from scratch ===
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)

            # Header
            page.draw_rect(fitz.Rect(0, 0, 595, 80), color=blue, fill=blue)
            page.insert_text(fitz.Point(180, 35), "AMDP", fontsize=28, fontname="hebo", color=white)
            page.insert_text(fitz.Point(180, 55), "MADE FOR INDUSTRY", fontsize=8, fontname="helv", color=(0.8, 0.8, 0.85))

            # Title
            y = 110
            title_rect = fitz.Rect(30, y, 565, y + 50)
            page.insert_textbox(title_rect, titre_fiche, fontsize=22, fontname="hebo", color=black, align=1)
            y += 50

            # Reference
            ref_line = f"Réf: {ref_clean}  |  Marque: {marque}"
            tw_ref = fitz.get_text_length(ref_line, fontsize=10, fontname="helv")
            page.insert_text(fitz.Point((595 - tw_ref) / 2, y), ref_line, fontsize=10, fontname="helv", color=gray)
            y += 15
            page.draw_line(fitz.Point(30, y), fitz.Point(565, y), color=red, width=1.5)
            y += 20

            # Image + description
            if product_image_data:
                try:
                    page.insert_image(fitz.Rect(35, y, 220, y + 180), stream=product_image_data, keep_proportion=True)
                    desc_rect = fitz.Rect(235, y, 565, y + 180)
                except Exception:
                    desc_rect = fitz.Rect(35, y, 565, y + 120)
            else:
                desc_rect = fitz.Rect(35, y, 565, y + 120)

            presentation_text = (desc_courte + "\n\n" + desc_longue[:300]) if desc_longue else desc_courte
            page.insert_textbox(desc_rect, presentation_text[:600], fontsize=10, fontname="helv", color=black, align=0)
            y += 200

            # Characteristics
            if carac_lines:
                page.insert_text(fitz.Point(30, y), "Caractéristiques techniques", fontsize=12, fontname="hebo", color=blue)
                y += 5
                page.draw_line(fitz.Point(30, y), fitz.Point(300, y), color=blue, width=1)
                y += 15
                for i, carac in enumerate(carac_lines[:15]):
                    if y > 760:
                        break
                    bg = light_gray if i % 2 == 0 else white
                    page.draw_rect(fitz.Rect(25, y - 2, 570, y + 15), color=bg, fill=bg)
                    if ":" in carac:
                        key, val = carac.split(":", 1)
                        page.insert_text(fitz.Point(30, y + 10), key.strip()[:45], fontsize=9, fontname="hebo", color=black)
                        page.insert_text(fitz.Point(250, y + 10), val.strip()[:55], fontsize=9, fontname="helv", color=gray)
                    else:
                        page.insert_text(fitz.Point(30, y + 10), carac[:80], fontsize=9, fontname="helv", color=black)
                    y += 18

            # Footer
            page.draw_rect(fitz.Rect(0, 802, 595, 842), color=blue, fill=blue)
            page.insert_text(fitz.Point(8, 822), "AMDP", fontsize=10, fontname="helv", color=white)
            page.insert_text(fitz.Point(4, 834), "MADE FOR INDUSTRY", fontsize=4, fontname="helv", color=(0.8, 0.8, 0.85))
            page.insert_text(fitz.Point(70, 818), "8 boulevard Georges-Marie Guynemer,", fontsize=7, fontname="helv", color=white)
            page.insert_text(fitz.Point(70, 830), "78210, Saint-Cyr-l'École, FRANCE", fontsize=7, fontname="helv", color=white)
            page.insert_text(fitz.Point(430, 818), "info@amdp.fr", fontsize=7, fontname="helv", color=white)
            page.insert_text(fitz.Point(430, 830), "www.amdp.shop", fontsize=7, fontname="helv", color=red)

            pdf_bytes = doc.tobytes()
            doc.close()

        return StreamingResponse(
            io_mod.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Fiche_{ref}.pdf"}
        )
    except Exception as e:
        import traceback
        print("ERROR fiche produit:", traceback.format_exc())
        return {"error": str(e)}