# utils/embedder.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 🔹 Qdrant Cloud configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "insurance_docs"

# 🔹 HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔹 Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# -------------------------
# Store Chunks in Qdrant
# -------------------------
def store_chunks_qdrant(chunks):
    """
    Stores document chunks in Qdrant Cloud collection.
    chunks: list of LangChain Document objects
    """
    if not chunks:
        raise ValueError("Chunks list is empty.")

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    print(f"📤 Uploading {len(texts)} chunks to Qdrant Cloud...")

    # ✅ Ensure collection exists (auto-create)
    if QDRANT_COLLECTION not in [col.name for col in qdrant_client.get_collections().collections]:
        print(f"⚙️ Creating Qdrant collection '{QDRANT_COLLECTION}'...")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=384,  # all-MiniLM-L6-v2 embedding dimension
                distance=Distance.COSINE
            )
        )

    # ✅ Push data to Qdrant
    Qdrant.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION
    )



    print("✅ Chunks stored successfully in Qdrant Cloud.")

# -------------------------
# Load Qdrant Collection
# -------------------------
def load_qdrant():
    """
    Loads existing Qdrant collection as a retriever.
    """
    print("📥 Loading Qdrant collection...")
    return Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embeddings
    )
