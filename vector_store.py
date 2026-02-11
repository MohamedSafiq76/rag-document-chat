"""
Vector Store Module
Handles Chroma DB operations: initialization, document storage, and semantic search.
Uses Chroma's built-in ONNX embedding (no PyTorch needed).
"""

import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"

# Chroma's built-in embedding — uses ONNX MiniLM-L6-v2 (no torch required)
_embedding_fn = DefaultEmbeddingFunction()


def _get_collection():
    """Return the Chroma collection, creating it if needed."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_documents(documents):
    """
    Embed and store a list of LangChain Documents into the Chroma DB.

    Parameters
    ----------
    documents : list[Document]
        Chunked documents from the ingestion module.
    """
    collection = _get_collection()

    ids = [f"doc_{i}_{hash(doc.page_content) % 100000}" for i, doc in enumerate(documents)]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Add in batches of 100 to avoid memory issues
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )


def search(query, k=5):
    """
    Perform semantic search and return the top-k most relevant document chunks.

    Parameters
    ----------
    query : str
        The user's question.
    k : int
        Number of results to return.

    Returns
    -------
    list[Document]
        The most relevant document chunks with metadata.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    # Don't request more results than exist
    k = min(k, count)

    results = collection.query(query_texts=[query], n_results=k)

    documents = []
    for i in range(len(results["documents"][0])):
        documents.append(Document(
            page_content=results["documents"][0][i],
            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
        ))
    return documents


def get_document_count():
    """Return the number of documents currently stored in the vector DB."""
    try:
        collection = _get_collection()
        return collection.count()
    except Exception:
        return 0


def clear():
    """Remove all documents from the Chroma DB collection."""
    try:
        collection = _get_collection()
        # Get all IDs and delete them — avoids file lock issues on Windows
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        pass
