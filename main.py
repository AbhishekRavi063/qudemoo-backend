


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
import logging
from contextlib import asynccontextmanager



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qu-demo-clipboardai.vercel.app",
        "https://qudemo-waiting-list-git-v2-clipboardai.vercel.app",
        "https://www.qudemo.com"
    ],
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


puzzle_VIDEO_URL_MAP = {
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

mixpanel_VIDEO_URL_MAP = {
    "downloaded_video_0.mp4": "https://youtu.be/c7SfDNFhD0E?si=J-OC5RCbIDFjdO2o",
    "downloaded_video_1.mp4": "https://youtu.be/ePIemj9gOgM?si=iB_3Xz60ZiEVKrcc",
    "downloaded_video_2.mp4": "https://youtu.be/0zdljj5xVjw?si=oE3iEzkrZnd8i7YR",
    "downloaded_video_3.mp4": "https://youtu.be/uiR1gU_JzDg?si=2XsCt2JDv4tjhpRp",
    "downloaded_video_4.mp4": "https://youtu.be/1KLbnVO_N_Y?si=Z6xk000W3-FdeHCC",
    "downloaded_video_5.mp4": "https://youtu.be/ocTKKaQe65o?si=HCZ9rlfI0XvAQMNH",
    "downloaded_video_6.mp4": "https://youtu.be/pzWpCH4jvJQ?si=jBcDKSLcy104W13t",
    "downloaded_video_7.mp4": "https://youtu.be/45ZBaJg-oe4?si=HtPCz1Ar6Xh4rz92",
    "downloaded_video_8.mp4": "https://youtu.be/KYiYvs-4YfI?si=A_jX7Sue7r80RKJN",
    "downloaded_video_9.mp4": "https://youtu.be/JfbyJuR3-Tg?si=eDk41a8KoIEweheY",
    "downloaded_video_10.mp4": "https://youtu.be/kbjkUeu8v3M?si=3x8RwpbsJW84ONMh",
    "downloaded_video_11.mp4": "https://youtu.be/XXAINxxuATo?si=FwzeQdbWfpbP5eKA",
    "downloaded_video_12.mp4": "https://youtu.be/DAzyfipugO0?si=XyfO0ncOng3VatCv",
    "downloaded_video_13.mp4": "https://youtu.be/fGOG_CDN3pA?si=XPZQE7WQ0czyfq5z",
    "downloaded_video_14.mp4": "https://youtu.be/8Pv6tmRfqr8?si=edHcdP-QQU4jFD1p",
    "downloaded_video_15.mp4": "https://youtu.be/xt1MXczb7io?si=XYFZ6samaGo2Ctsz",
    "downloaded_video_16.mp4": "https://youtu.be/7UJUE3EfKQg?si=9mzV8x6ckl7PYd6M",
    "downloaded_video_17.mp4": "https://youtu.be/lOdSRETdL-g?si=DBJaIK0NV71E__OV",
    "downloaded_video_18.mp4": "https://youtu.be/XRA9EUnd-c4?si=wTyw6pOdKuEqovhU",
    "downloaded_video_19.mp4": "https://youtu.be/1ierIlL_wQs?si=POusJXiNw14t7A24",
    "downloaded_video_20.mp4": "https://youtu.be/okaXAEqW59U?si=_RVYtkImWlzO0NM_",
    "downloaded_video_21.mp4": "https://youtu.be/9TN2OeYGN1I?si=WQuxIHXpMdGVgmlk",
    "downloaded_video_22.mp4": "https://youtu.be/hBZn3a8RSMw?si=aLC0xRs_WbpYL2Kf",
    "downloaded_video_23.mp4": "https://youtu.be/TbyKerzgxqM?si=6Wkoot57a8tjalMQ",
    "downloaded_video_24.mp4": "https://youtu.be/UYH5iueY5Js?si=m0lxljkGyAFyN-_g",
    "downloaded_video_25.mp4": "https://youtu.be/JwFzKlMP-Mc?si=fHQugqpPtvVqelxU",
    "downloaded_video_26.mp4": "https://youtu.be/4SdgUxYGX0c?si=vsjJI9GRZVx7a5cD",
    "downloaded_video_27.mp4": "https://youtu.be/snBXh_HHyPY?si=m_Uv5cfaGheSIdAp",
    "downloaded_video_28.mp4": "https://youtu.be/5hJBtqbtx9c?si=GD6wm6WFYzmNxDCD",
    "downloaded_video_29.mp4": "https://youtu.be/FnItIYNpBrM?si=Yqb4_euX6CZsnEbO"
}


# Input model
class BucketInput(BaseModel):
    source: str

VIDEO_URL_MAP = puzzle_VIDEO_URL_MAP




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

# def upload_faiss_index(local_path=FAISS_INDEX_PATH_LOCAL):
#     creds = get_credentials()
#     client = storage.Client(credentials=creds)
#     bucket = client.bucket(TRANSCRIPT_BUCKET)
#     blob = bucket.blob(FAISS_INDEX_PATH_GCS)
#     blob.upload_from_filename(local_path)

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
    content_bytes = blob.download_as_bytes()
    content = content_bytes.decode("utf-8", errors="replace")  # or "ignore"


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


# @app.on_event("startup")
# def startup_event():
#     global all_chunks, faiss_index
#     all_chunks = load_transcript_chunks()
#     print(f"✅ Loaded {len(all_chunks)} transcript chunks.")
#     faiss_index = load_faiss_index()
#     # load_faqs()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global all_chunks, faiss_index
    all_chunks = load_transcript_chunks()
    print(f"✅ Loaded {len(all_chunks)} transcript chunks.")
    faiss_index = load_faiss_index()
    yield  # You can place any cleanup logic after this line

app = FastAPI(lifespan=lifespan)

@app.post("/bucket")
async def set_bucket(bucket_input: BucketInput):
    global TRANSCRIPT_BUCKET, VIDEO_URL_MAP, all_chunks, faiss_index

    source = bucket_input.source.strip().lower()

    if source == "puzzle":
        TRANSCRIPT_BUCKET = "transcript_puzzle_v2"
        VIDEO_URL_MAP = puzzle_VIDEO_URL_MAP
    elif source == "mixpanel":
        TRANSCRIPT_BUCKET = "mixpanel_v1"
        VIDEO_URL_MAP = mixpanel_VIDEO_URL_MAP
    else:
        return {
            "status": "error",
            "message": f"Unknown source: {source}"
        }

    # ✅ Reload dependent resources
    all_chunks = load_transcript_chunks()
    faiss_index = load_faiss_index()
    print(f"🔁 Switching to bucket: {TRANSCRIPT_BUCKET}")  

    return {
        "status": "success",
        "transcript_bucket": TRANSCRIPT_BUCKET,
        "video_url_map": VIDEO_URL_MAP
    }



@app.post("/ask")
def ask_question(payload: Question):
    logger.info(f"📥 Received question: {payload.question}")

    try:
        q_embedding = openai.embeddings.create(
            input=[payload.question],
            model="text-embedding-3-small",
            timeout=15
        ).data[0].embedding
        logger.info("✅ Created embedding for the question.")
    except Exception as e:
        logger.error(f"❌ Failed to create question embedding: {e}")
        return {"error": "Failed to create question embedding."}

    # try:
    #     similarities = cosine_similarity(
    #         np.array([q_embedding]), FAQ_EMBEDDINGS
    #     )[0]

    #     best_idx = int(np.argmax(similarities))
    #     best_score = similarities[best_idx]
    #     logger.info(f"🔍 Best FAQ similarity score: {best_score:.3f}")

    #     if best_score > 0.85:
    #         logger.info("📚 Matched answer from FAQ.")
    #         return {
    #             "answer": FAQ_DATA[best_idx]["answer"],
    #             "sources": ["faq"],
    #             "video_url": None
    #         }
    # except Exception as e:
    #     logger.warning(f"⚠️ FAQ similarity check failed: {e}")

    # try:
    #     D, I = faiss_index.search(np.array([q_embedding], dtype="float32"), k=6)
    #     top_chunks = [all_chunks[i] for i in I[0]]
    #     logger.info(f"🔎 Retrieved top {len(top_chunks)} chunks from FAISS.")
    # except Exception as e:
    #     logger.error(f"❌ FAISS search failed: {e}")
    #     return {"error": f"FAISS search failed: {e}"}

    # try:
    #     rerank_prompt = f"Question: {payload.question}\n\nHere are the chunks:\n"
    #     for i, chunk in enumerate(top_chunks):
    #         snippet = chunk["text"][:500].strip().replace("\n", " ")
    #         rerank_prompt += f"{i+1}. [{chunk['type']}] {chunk.get('context','')}\n{snippet}\n\n"
    #     rerank_prompt += "Which chunk is most relevant to the question above? Just give the number."

    #     rerank_response = openai.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": rerank_prompt}],
    #         timeout=20
    #     )
    #     best_index = int(re.findall(r"\d+", rerank_response.choices[0].message.content)[0]) - 1
    #     best_chunk = top_chunks[best_index]
    #     logger.info(f"🏅 GPT-3.5-turbo reranked chunk #{best_index+1} as the most relevant.")
    # except Exception as e:
    #     best_chunk = top_chunks[0]
    #     logger.warning(f"⚠️ Reranking failed, falling back to top FAISS chunk: {e}")

    # try:
    #     context = "\n\n".join([
    #         f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks[:3]
    #     ])

       
    #     system_prompt = (
    #         "You are a product expert bot with full knowledge of Puzzle.io derived from video transcripts. "
    #         "Use clear, confident, and concise answers—no more than 700 characters. "
    #         "Use bullet points or short paragraphs if needed. Do not include inline citations like [source](...)."
    #     )
    #     user_prompt = f"Context:\n{context}\n\nQuestion: {payload.question}"

    #     completion = openai.chat.completions.create(
    #         model="gpt-4-turbo",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt},
    #         ],
    #         timeout=20
    #     )
    #     raw_answer = completion.choices[0].message.content
    #     logger.info("✅ Generated answer with GPT-4.")
    # except Exception as e:
    #     logger.error(f"❌ Failed to generate GPT-4 answer: {e}")
    #     return {"error": "Failed to generate answer."}

    # def strip_sources(text):
    #     return re.sub(r'\[source\]\([^)]+\)', '', text).strip()

    # def format_answer(text):
    #     text = re.sub(r'\s*[-•]\s+', r'\n• ', text)
    #     text = re.sub(r'\s*\d+\.\s+', lambda m: f"\n{m.group(0)}", text)
    #     return re.sub(r'\n+', '\n', text).strip()

    # raw_answer = strip_sources(raw_answer)
    # clean_answer = format_answer(raw_answer)

    # sources = [chunk["source"] for chunk in top_chunks]

    # def extract_time(url):
    #     match = re.search(r"[?&]t=(\d+)", url)
    #     return int(match.group(1)) if match else float("inf")

    # video_url = best_chunk["source"]

    # logger.info(f"📤 Returning final answer. Video URL: {video_url}")

    # return {
    #     "answer": clean_answer,
    #     "sources": sources,
    #     "video_url": video_url
    # }

    
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
        # best_chunk = top_chunks[best_index]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        best_chunk = top_chunks[0]

    try:
        context = "\n\n".join([
            f"{chunk['source']}: {chunk['text'][:500]}" for chunk in top_chunks
        ])

        # Dynamically determine project type based on TRANSCRIPT_BUCKET
        project_type = "the company"  # fallback
        if "puzzle" in TRANSCRIPT_BUCKET.lower():
            project_type = "Puzzle.io"
        elif "mixpanel" in TRANSCRIPT_BUCKET.lower():
            project_type = "Mixpanel"

        system_prompt = (
            f"You are a product expert bot with full knowledge of {project_type} derived from video transcripts. "
            "Use clear, confident, and concise answers—no more than 700 characters. "
            "Use bullet points or short paragraphs if needed. Do not include inline citations like [source](...)."
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
        print(f"❌ Answer generation failed: {e}")
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

    # Extract timestamp from URL and find the one with smallest time
    def extract_time(url):
        match = re.search(r"[?&]t=(\d+)", url)
        return int(match.group(1)) if match else float("inf")

    video_url = min(sources, key=extract_time, default=None)

    return {
        "answer": clean_answer,
        "sources": sources,
        "video_url": video_url
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway provides PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)