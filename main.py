import os
import json
import base64
import gdown
import fitz
import numpy as np
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configuration ──────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DRIVE_CATALOGUE  = "1TP1X8JQW02ujnV7CGC44tiBAr6ZjXXqM"
DRIVE_INDEX      = "1V6H-XKAh9RjbiD8aRttW3YhbSYdOswwk"
DRIVE_EMBEDDINGS = "1q4KI9oH4EuqsawJCAEIPnYU6HqAzf4rN"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PDF_LOCAL        = os.path.join(BASE_DIR, "catalogue.pdf")
INDEX_LOCAL      = os.path.join(BASE_DIR, "index_catalogue.json")
EMBEDDINGS_LOCAL = os.path.join(BASE_DIR, "embeddings_256.json")

client = OpenAI(api_key=OPENAI_API_KEY)

# Questions larges = synthèse texte uniquement (images trop coûteuses sur 15+ pages)
MOTS_QUESTION_LARGE = {
    "combien", "liste", "tous", "toutes", "types", "type", "gamme", "gammes",
    "catégorie", "categories", "quels", "quelles", "différents", "différentes",
    "existe", "disponibles", "disponible", "ensemble", "propose",
    "meilleures", "meilleurs", "recommandes", "recommandés", "comparaison"
}


# ── Téléchargement Google Drive ────────────────────────────────────────────────
def telecharger_drive(file_id: str, destination: str):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False, fuzzy=True)


def charger_fichiers():
    print("Chargement des fichiers...")
    if not os.path.exists(PDF_LOCAL):
        print("  Telechargement catalogue PDF...")
        telecharger_drive(DRIVE_CATALOGUE, PDF_LOCAL)
    if not os.path.exists(INDEX_LOCAL):
        print("  Telechargement index...")
        telecharger_drive(DRIVE_INDEX, INDEX_LOCAL)
    if not os.path.exists(EMBEDDINGS_LOCAL):
        print("  Telechargement embeddings...")
        telecharger_drive(DRIVE_EMBEDDINGS, EMBEDDINGS_LOCAL)

    print("  Chargement en memoire...")
    with open(EMBEDDINGS_LOCAL, "r", encoding="utf-8") as f:
        embeddings = np.array(json.load(f), dtype=np.float32)
    with open(INDEX_LOCAL, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if len(embeddings) != len(ids):
        raise ValueError(f"Mismatch : {len(embeddings)} embeddings vs {len(ids)} chunks dans l'index !")

    print(f"Pret ! {len(embeddings)} morceaux indexes.")
    return embeddings, ids


# ── Recherche ──────────────────────────────────────────────────────────────────
def traduire_query(question: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=60,
        messages=[
            {"role": "system", "content": "Translate to English. Reply with translation only."},
            {"role": "user", "content": question}
        ]
    )
    return r.choices[0].message.content.strip()


def recherche_semantique(query_en: str, embeddings, ids, n: int = 20):
    """Retourne les n meilleurs chunks par similarité cosinus. Clé = index du chunk."""
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_en],
        dimensions=256
    )
    q = np.array(r.data[0].embedding, dtype=np.float32)
    normes = np.linalg.norm(embeddings, axis=1)
    scores = np.dot(embeddings, q) / (normes * np.linalg.norm(q) + 1e-10)
    top = np.argsort(scores)[::-1][:n]
    return {int(i): float(scores[i]) for i in top}


def recherche_textuelle(question: str, ids, n: int = 20):
    """Recherche par mots-clés. Clé = index du chunk."""
    mots = [m.strip(".,;:?!()") for m in question.split() if len(m) >= 2]
    scores = {}
    for idx, item in enumerate(ids):
        texte = item["texte"].lower()
        score = sum(1 for m in mots if m.lower() in texte)
        if score > 0:
            scores[idx] = score
    top = sorted(scores, key=scores.get, reverse=True)[:n]
    max_score = max(scores.values()) if scores else 1
    return {i: scores[i] / max_score for i in top}


def recherche_hybride(question: str, query_en: str, embeddings, ids, n: int = 5):
    """
    Fusionne sémantique (60%) + textuelle (40%) par index de chunk.
    Retourne n chunks distincts triés par score.
    """
    sem = recherche_semantique(query_en, embeddings, ids, 20)
    txt = recherche_textuelle(question, ids, 20)

    tous_indices = set(sem.keys()) | set(txt.keys())
    fusionnes = []
    for idx in tous_indices:
        score = 0.6 * sem.get(idx, 0) + 0.4 * txt.get(idx, 0)
        fusionnes.append((idx, score))

    fusionnes.sort(key=lambda x: x[1], reverse=True)
    return [ids[idx] for idx, _ in fusionnes[:n]]


# ── Vision ─────────────────────────────────────────────────────────────────────
def page_en_image_base64(numero_page: int) -> str:
    doc = fitz.open(PDF_LOCAL)
    page = doc[numero_page - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    image_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(image_bytes).decode("utf-8")


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_TEXTE = """Tu es un assistant expert en outillage professionnel.
Le catalogue est en anglais. Tu réponds toujours en français.
À partir des extraits fournis, réponds à la question de manière directe.
Donne les références produits, noms et caractéristiques pertinentes.
Indique la page source entre parenthèses après chaque produit.
Pas d'introduction ni de conclusion inutile.
Si les extraits ne contiennent rien de pertinent, dis-le brièvement."""

SYSTEM_VISION = """Tu es un assistant expert en outillage professionnel.
Le catalogue est en anglais. Tu réponds toujours en français.
Analyse le texte ET les images/tableaux fournis pour répondre à la question.
Les tableaux contiennent des références exactes, dimensions (A, B, C...) et caractéristiques — exploite-les impérativement.
Donne les références produits exactes et leurs caractéristiques.
Indique la page source entre parenthèses.
Pas d'introduction ni de conclusion inutile.
Si les extraits ne contiennent rien de pertinent, dis-le brièvement."""


# ── Réponse finale ─────────────────────────────────────────────────────────────
def poser_question(embeddings, ids, question: str) -> str:
    large = any(m in question.lower().split() for m in MOTS_QUESTION_LARGE)
    query_en = traduire_query(question)

    if large:
        # Question large : texte uniquement, 12 chunks, gpt-4o-mini
        pertinents = recherche_hybride(question, query_en, embeddings, ids, n=12)
        if not pertinents:
            return "Aucune information trouvée dans le catalogue pour cette question."

        contexte = "\n\n".join([f"[Page {p['page']}]\n{p['texte']}" for p in pertinents])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_TEXTE},
                {"role": "user", "content": f"Extraits catalogue :\n\n{contexte}\n\nQuestion : {question}"}
            ]
        )

    else:
        # Question précise : 5 meilleurs chunks + images des pages uniques, gpt-4o
        pertinents = recherche_hybride(question, query_en, embeddings, ids, n=5)
        if not pertinents:
            return "Aucune information trouvée dans le catalogue pour cette question."

        contexte = "\n\n".join([f"[Page {p['page']}]\n{p['texte']}" for p in pertinents])

        # Pages uniques à visualiser (max 3 pour limiter les coûts)
        pages_vues = []
        for p in pertinents:
            if p["page"] not in pages_vues:
                pages_vues.append(p["page"])
            if len(pages_vues) >= 3:
                break

        contenu = [{"type": "text", "text": f"Extraits catalogue :\n\n{contexte}\n\nQuestion : {question}"}]
        for num_page in pages_vues:
            try:
                img_b64 = page_en_image_base64(num_page)
                contenu.append({"type": "text", "text": f"--- Image page {num_page} ---"})
                contenu.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "low"
                }})
            except Exception:
                pass

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_VISION},
                {"role": "user", "content": contenu}
            ]
        )

    return response.choices[0].message.content


# ── API FastAPI ────────────────────────────────────────────────────────────────
embeddings_global = None
ids_global = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings_global, ids_global
    try:
        embeddings_global, ids_global = charger_fichiers()
        print("Demarrage OK !")
    except Exception as e:
        print(f"ERREUR DEMARRAGE: {e}")
        raise
    yield


app = FastAPI(title="Chatbot Catalogue", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def accueil():
    return {"status": "API operationnelle"}


class Question(BaseModel):
    question: str


@app.post("/question")
def question(body: Question):
    reponse = poser_question(embeddings_global, ids_global, body.question)
    return {"question": body.question, "reponse": reponse}
