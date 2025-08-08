# main.py - Streamlit + FastAPI Backend for Insurance Assistant
from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.loader import load_and_chunk
from utils.embedder import store_chunks_qdrant, load_qdrant  # Qdrant methods
from utils.llm_reasoner import get_decision
import os, shutil, json, re
import google.generativeai as genai
from typing import Dict, Any
from dotenv import load_dotenv
import tempfile
from fastapi import Form
import urllib.parse


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- FastAPI App ---
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Query Parsing with Gemini
# -------------------------
def parse_query(query: str) -> Dict[str, Any]:
    try:
        prompt = f"""
        Extract the following fields from the query below:
        age, gender, procedure, location, policy_duration.
        Respond in JSON only.

        Query: {query}
        """
        response = model.generate_content(prompt)
        json_str = response.text.strip()

        # Extract JSON if wrapped in ```json``` fences
        if '```json' in json_str:
            json_str = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL).group(1)

        return json.loads(json_str)
    except Exception as e:
        return {"raw_query": query, "error": str(e)}

# -------------------------
# Upload Endpoint
# -------------------------
@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = load_and_chunk(file_path)
        store_chunks_qdrant(chunks)  # Store in Qdrant

        return {"message": "Uploaded and processed successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Query Endpoint
# -------------------------
@app.post("/query")
async def query_decision(query: str):
    parsed = parse_query(query)
    decision = get_decision(query)
    return {"parsed_query": parsed, "decision_result": decision}



@app.post("/bulk_query")
async def bulk_query(
    file: UploadFile = File(None),
    document_url: str = Form(None),
    questions: str = Form(...)
):
    try:
        # Save uploaded file or download from URL
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
        elif document_url:
            response = requests.get(document_url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download document from URL")

            parsed_url = urllib.parse.urlparse(document_url)
            ext = os.path.splitext(parsed_url.path)[1] or ".pdf"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
        else:
            raise HTTPException(status_code=400, detail="No file or document URL provided")

        chunks = load_and_chunk(tmp_path)
        store_chunks_qdrant(chunks)
        db = load_qdrant()

        questions_list = json.loads(questions) if questions.strip().startswith("[") else [questions]
        answers = []

        for question in questions_list:
            docs = db.similarity_search(question, k=4)

            def safe_metadata(md):
                safe_md = {}
                for k, v in md.items():
                    try:
                        json.dumps(v)
                        safe_md[k] = v
                    except Exception:
                        safe_md[k] = str(v)
                return safe_md

            context = "\n\n".join([
                f"Document: {safe_metadata(d.metadata).get('source', 'unknown')} | Page: {safe_metadata(d.metadata).get('page', '-')}\nClause: {d.page_content}"
                for d in docs
            ])

            prompt = (
                f"Answer the question using the given clauses.\n\n"
                f"Question: {question}\n\n"
                f"Clauses:\n{context}\n\n"
                "Answer in one short sentence."
            )
            answer = model.generate_content(prompt).text.strip()
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


