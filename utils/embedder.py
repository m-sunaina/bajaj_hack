import os
from dotenv import load_dotenv

# Load local .env only for local dev (Render injects env vars)
if os.getenv("RENDER") != "true":
    load_dotenv()

# Qdrant / embeddings config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "insurance_docs"

# singletons (cached)
_embeddings_instance = None
_qdrant_client_instance = None
_qdrant_vectorstore_instance = None


def get_embeddings():
    """Get or initialize HuggingFace embeddings (singleton)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        # import inside function to avoid import-time cost
        from langchain_huggingface import HuggingFaceEmbeddings
        print("⚡ Loading HuggingFace embeddings model...")
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance


def get_qdrant_client():
    """Get or initialize Qdrant client (singleton)."""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        from qdrant_client import QdrantClient
        print("⚡ Connecting to Qdrant Cloud...")
        _qdrant_client_instance = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    return _qdrant_client_instance


def get_qdrant_vectorstore():
    """Get or initialize a LangChain Qdrant vectorstore (singleton)."""
    global _qdrant_vectorstore_instance
    if _qdrant_vectorstore_instance is None:
        from langchain_qdrant import Qdrant
        print("⚡ Initializing Qdrant vectorstore wrapper...")
        _qdrant_vectorstore_instance = Qdrant(
            client=get_qdrant_client(),
            collection_name=QDRANT_COLLECTION,
            embeddings=get_embeddings()
        )
    return _qdrant_vectorstore_instance


def store_chunks_qdrant(chunks):
    """
    Stores document chunks in Qdrant Cloud collection.
    chunks: list of LangChain Document objects
    """
    if not chunks:
        raise ValueError("Chunks list is empty.")

    from qdrant_client.http.models import VectorParams, Distance

    embeddings = get_embeddings()
    qdrant_client = get_qdrant_client()
    vectorstore = get_qdrant_vectorstore()

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    print(f"📤 Uploading {len(texts)} chunks to Qdrant Cloud...")

    # Ensure collection exists
    try:
        existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    except Exception as e:
        # If client.get_collections() fails, still proceed to vectorstore which will raise a clearer error
        print(f"⚠️ Could not list collections: {e}")
        existing_collections = []

    if QDRANT_COLLECTION not in existing_collections:
        print(f"⚙️ Creating Qdrant collection '{QDRANT_COLLECTION}'...")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

    # add_texts uses the existing vectorstore and embeddings
    try:
        # prefer add_texts when available to avoid re-initializing internals
        vectorstore.add_texts(texts=texts, metadatas=metadatas)
    except Exception:
        # fallback to from_texts if needed by the current langchain version
        from langchain_qdrant import Qdrant
        Qdrant.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION
        )

    print("✅ Chunks stored successfully in Qdrant Cloud.")


def load_qdrant():
    """
    Loads existing Qdrant collection as a retriever (cached).
    """
    print("📥 Loading Qdrant collection (cached wrapper)...")
    return get_qdrant_vectorstore()


# Optional: try a safe preload on import, but don't crash if it fails.
# This attempts to make first request faster on stable environments.
try:
    # Keep this light — it will initialize clients + embeddings in normal environments,
    # but if the environment is constrained this will print an error and continue.
    get_qdrant_client()
    # Do NOT force embeddings load here to avoid import-time delays on render;
    # the startup warm-up will trigger it if desired.
    print("ℹ️ Qdrant client initialized (embedder module).")
except Exception as e:
    print(f"⚠️ embedder pre-init skipped: {e}")
