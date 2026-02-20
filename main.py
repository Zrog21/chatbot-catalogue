import io
import os
import json
import base64
import requests
import pdfplumber
import fitz
import numpy as np
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DRIVE_CATALOGUE  = "1TP1X8JQW02ujnV7CGC44tiBAr6ZjXXqM"
DRIVE_INDEX      = "1wBKObHcLEN6Pg39MxUDFACjQqcZupZ2P"
DRIVE_PARTS      = [
    "169AqzqPQI-u7oGlBngZbXF0g6Qy1L661",
    "1v8NCmdQzo5AbUUgOWVH-ntR61bPuR_lj",
    "1tpU2aHW1EdbJqsc1YLb9OsMAV8b7MQNc",
    "1UCO1GR6h5pX4ADzVcUIaBV3pPOi6Hfz7",
]

PDF_LOCAL   = "/tmp/catalogue.pdf"
INDEX_LOCAL = "/tmp/index.json"

client = OpenAI(api_key=OPENAI_API_KEY)

MOTS_QUESTION_LARGE = {
    "combien", "liste", "tous", "toutes", "types", "type", "gamme",
    "gammes", "catégorie", "categories", "quels", "quelles", "différents",
    "différentes", "existe", "disponibles", "disponible", "ensemble", "propose"
}


def telecharger_drive(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token:
        response = session.get(url + f"&confirm={token}", stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)


def charger_fichiers():
    print("Chargement des fichiers depuis Google Drive...")

    if not os.path.exists(PDF_LOCAL):
        print("  Téléchargement catalogue PDF...")
        telecharger_drive(DRIVE_CATALOGUE, PDF_LOCAL)

    if not os.path.exists(INDEX_LOCAL):
        print("  Téléchargement index...")
        telecharger_drive(DRIVE_INDEX, INDEX_LOCAL)

    # Téléchargement et fusion des 4 parties d'embeddings
    tous_embeddings = []
    for i, drive_id in enumerate(DRIVE_PARTS, start=1):
        part_path = f"/tmp/embeddings_part{i}.json"
        if not os.path.exists(part_path):
            print(f"  Téléchargement embeddings partie {i}/4...")
            telecharger_drive(drive_id, part_path)
        print(f"  Chargement partie {i}/4...")
        with open(part_path, "r", encoding="utf-8") as f:
            tous_embeddings.extend(json.load(f))

    embeddings = np.array(tous_embeddings, dtype=np.float32)

    with open(INDEX_LOCAL, "r", encoding="utf-8") as f:
        ids = json.load(f)

    print(f"Prêt ! {len(embeddings)} morceaux indexés.")
    return embeddings, ids


def chercher_semantique(question, embeddings, ids, n=5):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )
    q_embedding = np.array(response.data[0].embedding, dtype=np.float32)
    normes = np.linalg.norm(embeddings, axis=1)
    similarites = np.dot(embeddings, q_embedding) / (normes * np.linalg.norm(q_embedding) + 1e-10)
    meilleurs = np.argsort(similarites)[::-1][:n]
    return [ids[i] for i in meilleurs]


def est_question_large(question):
    mots = question.lower().split()
    return any(m in MOTS_QUESTION_LARGE for m in mots)


def page_en_image_base64(numero_page):
    doc = fitz.open(PDF_LOCAL)
    page = doc[numero_page - 1]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    image_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(image_bytes).decode("utf-8")


def poser_question(embeddings, ids, question):
    large = est_question_large(question)
    nb_resultats = 15 if large else 4
    pertinents = chercher_semantique(question, embeddings, ids, nb_resultats)
    if not pertinents:
        return "Je n'ai pas trouvé d'information sur ce sujet dans le catalogue."

    contexte = "\n\n".join([f"[Page {p['page']}]\n{p['texte']}" for p in pertinents])

    if large:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "Tu es un assistant expert en outillage professionnel. "
                    "Réponds aux questions générales en synthétisant toutes les informations pertinentes. "
                    "Indique les numéros de pages sources."
                )},
                {"role": "user", "content": f"Extraits du catalogue :\n\n{contexte}\n\nQuestion : {question}"}
            ]
        )
    else:
        images = []
        pages_vues = set()
        for p in pertinents:
            if p["page"] not in pages_vues:
                try:
                    img_b64 = page_en_image_base64(p["page"])
                    images.append({"page": p["page"], "b64": img_b64})
                    pages_vues.add(p["page"])
                except Exception:
                    pass

        contenu = [{"type": "text", "text": f"Extraits du catalogue :\n\n{contexte}\n\nQuestion : {question}"}]
        for img in images:
            contenu.append({"type": "text", "text": f"Image page {img['page']} :"})
            contenu.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['b64']}", "detail": "high"}})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Tu es un assistant expert en outillage professionnel. "
                    "Analyse le texte et les images. "
                    "Pour les dimensions (A, B, C...), explique concrètement à quoi elles correspondent. "
                    "Indique le numéro de page source. Sois précis et concis."
                )},
                {"role": "user", "content": contenu}
            ]
        )

    return response.choices[0].message.content


# API FastAPI
app = FastAPI(title="Chatbot Catalogue")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

embeddings_global = None
ids_global = None

@app.on_event("startup")
async def startup():
    global embeddings_global, ids_global
    try:
        embeddings_global, ids_global = charger_fichiers()
        print("Démarrage OK !")
    except Exception as e:
        print(f"ERREUR DEMARRAGE: {e}")
        raise

@app.get("/")
def accueil():
    return {"status": "API opérationnelle"}

class Question(BaseModel):
    question: str

@app.post("/question")
def question(body: Question):
    reponse = poser_question(embeddings_global, ids_global, body.question)
    return {"question": body.question, "reponse": reponse}
