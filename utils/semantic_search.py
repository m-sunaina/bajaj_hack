# semantic_searcher.py
from utils.embedder import load_qdrant

# Load the Qdrant collection
db = load_qdrant()

def search(query, top_k=3):
    """
    Perform semantic search on Qdrant vector store.
    Args:
        query (str): User query
        top_k (int): Number of results to return
    Returns:
        list of dicts with text, page, and source
    """
    docs = db.similarity_search(query, k=top_k)
    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "unknown")
        })
    return results
