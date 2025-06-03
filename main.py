from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import faiss
import numpy as np
import json
import os
import io
import re
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
from PyPDF2 import PdfReader

load_dotenv()

app = FastAPI()

# ✅ CORS for frontend - adjust origins as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],  # Change to your frontend URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Google Cloud Storage bucket info
TRANSCRIPT_BUCKET = "transcript_puzzle"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"
FAISS_INDEX_PATH_GCS = "faiss_indexes/faiss_index.bin"

PDF_BUCKET = "puzzle_io"
PDF_FOLDER = "pdf/"

VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/ZAGxqOT2l2U?si=uSwgsYfcqKMxAWGc",
    "downloaded_video_1.mp4": "https://youtu.be/ZAGxqOT2l2U?si=DJ0JsvvIBIz19cJ1",
    "downloaded_video_2.mp4": "https://youtu.be/_zRaJOF-trE?si=7ob6ZbLED2butzfa",
    "downloaded_video_3.mp4": "https://youtu.be/opV4Tmgepno?si=-9aHDmOvNeQbLVDY",
    "downloaded_video_4.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=r3NYHGhDWE63puzm",
    "downloaded_video_5.mp4": "https://youtu.be/A_IVog6Vs3I?si=xUovSgHHxrj8jPLc",
    "downloaded_video_6.mp4": "https://youtu.be/Em8ixilyoEo?si=UVWgS9SOccmpytRP",
    "downloaded_video_7.mp4": "https://youtu.be/sIun13utbI4?si=89bQAHXd_KQ0opzE",
    "downloaded_video_8.mp4": "https://youtu.be/-6aSKEs94cs?si=ne1vxH5NC6VG0Cuu",
    "downloaded_video_9.mp4": "https://youtu.be/Dd2FxrAQQtI?si=WIr9qZwJkShqNNem",
    "downloaded_video_10.mp4": "https://youtu.be/7XivT1Ts2jU?si=UBhpiCKH9d4lSgRF",
    "downloaded_video_11.mp4": "https://youtu.be/Tt8ucqPwfzM?si=CJqwRIkxFZhI8oGn",
    "downloaded_video_12.mp4": "https://youtu.be/tbupLhuf-yo?si=DdI4JM1mu3N5e1wU"
}

def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)

def download_faiss_index(local_path="faiss_index.bin"):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    if not blob.exists():
        raise RuntimeError(f"FAISS index file {FAISS_INDEX_PATH_GCS} not found in bucket {TRANSCRIPT_BUCKET}")
    blob.download_to_filename(local_path)
    print(f"✅ Downloaded FAISS index from GCS to {local_path}")

def load_faiss_index(local_path="faiss_index.bin"):
    if not os.path.exists(local_path):
        download_faiss_index(local_path)
    index = faiss.read_index(local_path)
    print(f"✅ Loaded FAISS index from {local_path}")
    return index

def load_transcript_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(TRANSCRIPT_JSON_PATH)
    content = blob.download_as_text()
    chunks = json.loads(content)
    enriched_chunks = []
    for chunk in chunks:
        source = chunk.get("source", "")
        m = re.match(r"(.+\.mp4) \[(\d{2}):(\d{2}):(\d{2}),", source)
        if m:
            filename = m.group(1)
            h, mm, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
            seconds = h * 3600 + mm * 60 + s
            yt_url = VIDEO_URL_MAP.get(filename)
            if yt_url:
                if '?' in yt_url:
                    enriched_source = f"{yt_url}&t={seconds}"
                else:
                    enriched_source = f"{yt_url}?t={seconds}"
                enriched_chunks.append({"source": enriched_source, "text": chunk["text"]})
                continue
        enriched_chunks.append(chunk)
    return enriched_chunks

def load_pdf_chunks():
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(PDF_BUCKET)
    blobs = client.list_blobs(PDF_BUCKET, prefix=PDF_FOLDER)
    chunks = []
    skipped = 0
    for blob in blobs:
        if not blob.name.lower().endswith(".pdf"):
            continue
        try:
            content = blob.download_as_bytes()
            reader = PdfReader(io.BytesIO(content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "source": f"{os.path.basename(blob.name)} (page {i+1})",
                        "text": text.strip()
                    })
        except Exception as e:
            skipped += 1
            print(f"⚠️ Skipped {blob.name}: {e}")
    print(f"✅ Loaded {len(chunks)} PDF chunks. Skipped {skipped} PDFs.")
    return chunks

@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    print("🚀 Starting up backend...")

    # Load transcript + PDF chunks
    transcript_chunks = load_transcript_chunks()
    pdf_chunks = load_pdf_chunks()
    all_chunks = transcript_chunks + pdf_chunks
    print(f"✅ Loaded {len(all_chunks)} chunks (transcripts + PDFs)")

    # Load FAISS index from file downloaded from GCS
    faiss_index = load_faiss_index()
    print("✅ FAISS index loaded successfully on startup.")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Question):
    print(f"🔍 Received question: {payload.question}")

    try:
        q_embedding = openai.embeddings.create(
            input=[payload.question], model="text-embedding-3-small"
        ).data[0].embedding
        print("✅ Created question embedding.")
    except Exception as e:
        print(f"❌ Error creating question embedding: {e}")
        return {"error": "Failed to create question embedding."}

    try:
        D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=5)
        top_chunks = [all_chunks[i] for i in I[0]]
        print(f"✅ Retrieved top {len(top_chunks)} chunks from FAISS.")
    except Exception as e:
        print(f"❌ FAISS search failed: {e}")
        return {"error": "Failed to search for relevant chunks."}

    context = "\n\n".join([f"{chunk['source']}: {chunk['text']}" for chunk in top_chunks])

    system_prompt = (
        "You are a concise, knowledgeable, and helpful assistant. Your task is to provide clear and accurate answers based on the context provided. "
        "Always prioritize quality and relevance over length—focus only on the most important details. "
        "If the information is derived from a video or PDF, include a citation with a timestamp (for videos) or page number (for PDFs). "
        "Present your answers in a well-organized and easy-to-understand format. Use paragraphs or bullet points if necessary to improve readability and structure."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw_answer = completion.choices[0].message.content
        print("✅ Received response from OpenAI chat completion.")
    except Exception as e:
        print(f"❌ OpenAI chat completion failed: {e}")
        return {"error": "Failed to generate answer."}

    clean_answer = ' '.join(raw_answer.split())

    first_video_source = None
    for chunk in top_chunks:
        if chunk["source"].startswith("http"):
            first_video_source = chunk["source"]
            break

    return {
        "answer": clean_answer,
        "sources": [chunk["source"] for chunk in top_chunks],
        "video_url": first_video_source
    }
