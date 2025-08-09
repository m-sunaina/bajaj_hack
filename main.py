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
from dotenv import load_dotenv
import google.generativeai as genai
from utils.loader import load_and_chunk
from utils.embedder import store_chunks_qdrant, load_qdrant

# Load env variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()
router = APIRouter(prefix="/api/v1")

api_key_header = APIKeyHeader(name="Authorization")

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@router.post("/hackrx/run")
async def hackrx_run(payload: HackRxRequest, authorization: str = Security(api_key_header)):
    if not authorization or not authorization.startswith("Bearer ") or authorization[7:] != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")

    try:
        resp = requests.get(payload.documents, timeout=30)
        resp.raise_for_status()

        url_path = urlparse(payload.documents).path
        suffix = os.path.splitext(url_path)[1] or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        chunks = load_and_chunk(tmp_path)
        store_chunks_qdrant(chunks)
        db = load_qdrant()

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
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass

app.include_router(router)

# Add global security scheme to OpenAPI for Swagger UI "Authorize" button
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
