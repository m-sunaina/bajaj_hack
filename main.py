from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRouter
from pydantic import BaseModel
from typing import List
import os
import requests
import tempfile
from urllib.parse import urlparse
import asyncio

# Local .env only for local dev
if os.getenv("RENDER") != "true":
    from dotenv import load_dotenv
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("API_KEY")

app = FastAPI()
router = APIRouter(prefix="/api/v1")

api_key_header = APIKeyHeader(name="Authorization")


class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]


# --- Warmup: run non-blocking background task after startup ---
@app.on_event("startup")
async def startup_warmup():
    """
    Attempt to warm embeddings in background so the first user request is faster.
    This does not block app startup (important to avoid Render timeout).
    """
    async def _warm_embeddings():
        try:
            # Import and call embed_query in an executor to avoid blocking the event loop
            from utils import embedder
            loop = asyncio.get_event_loop()
            # run small embedding call in threadpool
            await loop.run_in_executor(None, lambda: embedder.get_embeddings().embed_query("warmup"))
            print("✅ Background: embeddings warmup complete")
        except Exception as e:
            print(f"⚠️ Background: embeddings warmup failed: {e}")

    # Launch background warmup, but don't await it here
    asyncio.create_task(_warm_embeddings())


def get_gemini_model():
    """
    Lazily create / cache the Gemini model inside google.generativeai.
    We keep logic same as before but avoid doing this at import time.
    """
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


@router.post("/hackrx/run")
async def hackrx_run(payload: HackRxRequest, authorization: str = Security(api_key_header)):
    # Auth check
    if not authorization or not authorization.startswith("Bearer ") or authorization[7:] != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")

    try:
        # Download document
        resp = requests.get(payload.documents, timeout=30)
        resp.raise_for_status()

        # Save temp file
        url_path = urlparse(payload.documents).path
        suffix = os.path.splitext(url_path)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Lazy import processing functions (they handle their own lazy loads)
        from utils.loader import load_and_chunk
        from utils.embedder import store_chunks_qdrant, load_qdrant

        # Process and store chunks
        chunks = load_and_chunk(tmp_path)
        store_chunks_qdrant(chunks)
        db = load_qdrant()

        # Lazy load Gemini model
        model = get_gemini_model()

        answers = []
        for question in payload.questions:
            docs = db.similarity_search(question, k=4)
            context = "\n\n".join([f"Document Clause: {d.page_content}" for d in docs])
            prompt = (
                f"Answer this insurance policy question accurately and concisely:\n"
                f"Question: {question}\n"
                f"Relevant Policy Clauses:\n{context}\n"
                "Answer in one clear sentence under 30 words."
            )
            try:
                answer = model.generate_content(prompt).text.strip()
                answers.append(answer)
            except Exception as e:
                answers.append(f"Error generating answer: {str(e)}")

        return {"answers": answers}

    except requests.RequestException as e:
        raise HTTPException(400, f"Failed to download document: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        try:
            if "tmp_path" in locals():
                os.remove(tmp_path)
        except Exception:
            pass


app.include_router(router)


# Swagger Bearer Auth for docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="HackRx API",
        version="1.0.0",
        description="API with Bearer authentication",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
