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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qu-demo-clipboardai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

TRANSCRIPT_BUCKET = "transcript_puzzle_v2"
TRANSCRIPT_JSON_PATH = "transcripts/transcript_chunks.json"
FAISS_INDEX_PATH_LOCAL = "faiss_index.bin"
FAISS_INDEX_PATH_GCS = "faiss_indexes/faiss_index.bin"
FAQ_CSV_PATH = "csv/faq.csv"

VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/ZAGxqOT2l2U?si=isr_vVKKMLZoIzjn",
    "downloaded_video_1.mp4": "https://youtu.be/_zRaJOF-trE?si=mfvQvaGmVppQ1-X3",
    "downloaded_video_2.mp4": "https://youtu.be/o1ReLrUYPfY?si=BL4szpBzHZ0vImy3",
    "downloaded_video_3.mp4": "https://youtu.be/wR_JMXuUOpk?si=yvu3r2t_CkiEoaWR",
    "downloaded_video_4.mp4": "https://youtu.be/WA4N_3Fdk2A?si=2sCTGiHyiK0K1Klh",
    "downloaded_video_5.mp4": "https://youtu.be/q9cbCws782M?si=pz-L8VI3JzXUaXW4",
    "downloaded_video_6.mp4": "https://youtu.be/opV4Tmgepno?si=AD3H2kVjaHb7Zdbx",
    "downloaded_video_7.mp4": "https://youtu.be/q2Rb2ZR5eyw?si=c1al2NLZ9d5QdUpE",
    "downloaded_video_8.mp4": "https://youtu.be/TzFIfvtL5mk?si=5BMTCquJzcb7b9XC",
    "downloaded_video_9.mp4": "https://youtu.be/sIun13utbI4?si=PFRRrRhFGIR0O-8y",
    "downloaded_video_10.mp4": "https://youtu.be/-6aSKEs94cs?si=3bzK7q1Oez4UM2RK",
    "downloaded_video_11.mp4": "https://youtu.be/Dd2FxrAQQtI?si=6emu45SBBU1Pgtta",
    "downloaded_video_12.mp4": "https://youtu.be/7XivT1Ts2jU?si=e8GiyxunOhAo_fN_",
    "downloaded_video_13.mp4": "https://youtu.be/Tt8ucqPwfzM?si=bZ205FSJXypMh_qm",
    "downloaded_video_14.mp4": "https://youtu.be/tbupLhuf-yo?si=qoPQ4f0SuzTvCNtb",
    "downloaded_video_15.mp4": "https://youtu.be/Em8ixilyoEo?si=agoA4zJVMbrlXRMk",
    "downloaded_video_16.mp4": "https://youtu.be/A_IVog6Vs3I?si=6PJ71r6CcWkN2b_q",
    "downloaded_video_17.mp4": "https://youtu.be/tF0uoicP9Q0?si=Kg8i6D66zovXXCg1",
    "downloaded_video_18.mp4": "https://youtu.be/1EELDkH9tC8?si=DUf6fgdfytdljlTq",
    "downloaded_video_19.mp4": "https://youtu.be/_TfLvzLrCXA?si=hGPdoz1XpElCerXo",
    "downloaded_video_20.mp4": "https://youtu.be/vvmsA_EvPJA?si=0xNK4S3_DiXW92MH"
}


class Question(BaseModel):
    question: str

FAQ_EMBEDDINGS = []
FAQ_DATA = []

def get_credentials():
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.exists(key_path):
        raise RuntimeError("Valid GOOGLE_APPLICATION_CREDENTIALS path is required")
    return service_account.Credentials.from_service_account_file(key_path)

def download_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    if not blob.exists():
        raise RuntimeError(f"FAISS index {FAISS_INDEX_PATH_GCS} not found in bucket {TRANSCRIPT_BUCKET}")
    blob.download_to_filename(local_path)

def upload_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAISS_INDEX_PATH_GCS)
    blob.upload_from_filename(local_path)

def load_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
    if not os.path.exists(local_path):
        download_faiss_index(local_path)
    return faiss.read_index(local_path)

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
            filename, h, m_, s = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            seconds = h * 3600 + m_ * 60 + s
            yt_url = VIDEO_URL_MAP.get(filename)
            if yt_url:
                enriched_source = f"{yt_url}&t={seconds}" if "?" in yt_url else f"{yt_url}?t={seconds}"
                enriched_chunks.append({
                    "source": enriched_source,
                    "text": chunk["text"],
                    "type": "video",
                    "context": chunk.get("context", "")
                })
    return enriched_chunks

def load_faqs():
    global FAQ_EMBEDDINGS, FAQ_DATA

    creds = get_credentials()
    client = storage.Client(credentials=creds)
    bucket = client.bucket(TRANSCRIPT_BUCKET)
    blob = bucket.blob(FAQ_CSV_PATH)
    content = blob.download_as_text()

    df = pd.read_csv(io.StringIO(content))
    df.columns = df.columns.str.strip().str.lower()  # Normalize headers ✅

    FAQ_DATA = df.to_dict(orient="records")

    # Embed FAQ questions
    questions = [entry["question"] for entry in FAQ_DATA]
    embeddings = []
    for i in range(0, len(questions), 10):
        resp = openai.embeddings.create(
            input=questions[i:i+10],
            model="text-embedding-3-small",
            timeout=20
        )
        batch_embeddings = [e.embedding for e in resp.data]
        embeddings.extend(batch_embeddings)

    FAQ_EMBEDDINGS = np.array(embeddings).astype("float32")
    print(f"✅ Loaded {len(FAQ_DATA)} FAQ entries.")


@app.on_event("startup")
def startup_event():
    global all_chunks, faiss_index
    all_chunks = load_transcript_chunks()
    faiss_index = load_faiss_index()
    load_faqs()

@app.post("/ask")
def ask_question(payload: Question):
    try:
        q_embedding = openai.embeddings.create(
            input=[payload.question],
            model="text-embedding-3-small",
            timeout=15
        ).data[0].embedding
    except Exception as e:
        return {"error": "Failed to create question embedding."}

    try:
        similarities = cosine_similarity(
            np.array([q_embedding]), FAQ_EMBEDDINGS
        )[0]

        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score > 0.85:
            return {
                "answer": FAQ_DATA[best_idx]["answer"],
                "sources": ["faq"],
                "video_url": None
            }
    except Exception as e:
        pass

    try:
        D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=6)
        top_chunks = [all_chunks[i] for i in I[0]]
    except Exception as e:
        return {"error": f"FAISS search failed: {e}"}

    try:
        rerank_prompt = f"Question: {payload.question}\n\nHere are the chunks:\n"
        for i, chunk in enumerate(top_chunks):
            snippet = chunk["text"][:500].strip().replace("\n", " ")
            rerank_prompt += f"{i+1}. [{chunk['type']}] {chunk.get('context','')}\n{snippet}\n\n"
        rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."

        rerank_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rerank_prompt}],
            timeout=20
        )
        best_index = int(re.findall(r"\d+", rerank_response.choices[0].message.content)[0]) - 1
        best_chunk = top_chunks[best_index]
    except Exception as e:
        best_chunk = top_chunks[0]

    try:
        context = "\n\n".join([
            f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks
        ])

        system_prompt = (
                "You are a product expert bot with deep knowledge of Puzzle.io, strictly based on video transcripts and FAQs. "
                "Always prioritize and synthesize content from video transcripts. Use FAQs only to supplement when necessary. "
                "If the answer cannot be found in these sources, respond with 'Not mentioned in the videos or FAQs.' "
                "Do not guess or use external knowledge. "
                "Respond with clarity, confidence, and conciseness. Keep answers under 700 characters. "
                "Use bullet points or short paragraphs. Never hallucinate or provide information not explicitly present in the source material."
            )




        user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=20
        )
        raw_answer = completion.choices[0].message.content
    except Exception as e:
        return {"error": "Failed to generate answer."}

    def strip_sources(text):
        return re.sub(r'\[source\]\([^)]+\)', '', text).strip()

    def format_answer(text):
        text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
        text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
        return re.sub(r'\n+', '\n', text).strip()

    raw_answer = strip_sources(raw_answer)
    clean_answer = format_answer(raw_answer)

    sources = [chunk["source"] for chunk in top_chunks]

    def extract_time(url):
        match = re.search(r"[?&]t=(\d+)", url)
        return int(match.group(1)) if match else float("inf")

    video_url = min(sources, key=extract_time, default=None)

    return {
        "answer": clean_answer,
        "sources": sources,
        "video_url": video_url
    }
