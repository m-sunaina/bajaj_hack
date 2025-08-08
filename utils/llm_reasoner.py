import json
import re
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

# ======================
# QDRANT CONFIG
# ======================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "insurance_docs"

# ======================
# GEMINI CONFIG
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# ======================
# Load Qdrant Vector Store
# ======================
def load_qdrant():
    # Init embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Init Qdrant client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Connect as LangChain VectorStore
    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings
    )
    return vectorstore


# Load once
db = load_qdrant()


# ======================
# Search
# ======================
def search(query, top_k=3):
    return db.similarity_search(query, k=top_k)


# ======================
# Prompt Builder
# ======================
def build_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"Document: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}\nClause: {doc.page_content}"
        for doc in retrieved_docs
    ])

    return f"""
You are a health insurance claim reasoning assistant.

Based on the user query and the clauses extracted from health insurance policy documents, return a JSON object with the following fields:

- decision: "approved" or "rejected"
- amount: a numeric value (in INR) that indicates how much the insurance will cover (do not return null if any amount is mentioned in the clauses)
- justification: a list of matching clauses with their document name and page number

If no amount is mentioned at all, then and only then return amount as null.

User Query:
\"\"\"{query}\"\"\" 

Policy Clauses:
{context}

Strictly return a valid JSON object with no extra text or formatting.
"""


# ======================
# Fallback: Extract amount manually
# ======================
def extract_amount_from_clauses(retrieved_docs):
    pattern = r'(?:INR|Rs\.?|₹)\s?[\d,]+(?:\.\d+)?'
    for doc in retrieved_docs:
        match = re.search(pattern, doc.page_content)
        if match:
            num_str = match.group(0)
            cleaned = re.sub(r'[^\d.]', '', num_str)  # remove INR, commas, symbols
            try:
                return float(cleaned)
            except:
                continue
    return None


# ======================
# Main LLM Reasoner
# ======================
def get_decision(query, top_k=3):
    retrieved_docs = search(query, top_k)
    prompt = build_prompt(query, retrieved_docs)

    response = model.generate_content(prompt, request_options={"timeout": 60})  # Increased timeout
    raw_output = response.text.strip()

    # Clean JSON formatting
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("` \n")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:].strip()

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        print("❌ [ERROR] Gemini returned malformed JSON:")
        print(raw_output)
        return {}

    # Fallback for amount
    if result.get("amount") is None:
        fallback_amount = extract_amount_from_clauses(retrieved_docs)
        result["amount"] = fallback_amount

    return result


# ======================
# Test Run
# ======================
if __name__ == "__main__":
    query = "Patient hospitalized for 5 days due to dengue fever, what amount is covered?"
    decision = get_decision(query)
    print(decision)
