"""
Index medical literature into ChromaDB vector store.
Uses sentence-transformers for embeddings.

Usage:
    from rag.indexer import build_index
    collection = build_index()
"""
import os
import logging

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db")


def build_index(force_rebuild=False):
    """Build or load ChromaDB vector store from medical literature."""
    import chromadb
    from rag.documents import THYROID_LITERATURE

    os.makedirs(DB_PATH, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
    except Exception:
        client = chromadb.EphemeralClient()

    collection_name = "thyroid_literature"

    existing_names = [c.name for c in client.list_collections()]

    if collection_name in existing_names and not force_rebuild:
        collection = client.get_collection(collection_name)
        if collection.count() >= len(THYROID_LITERATURE):
            log.info(f"ChromaDB loaded: {collection.count()} documents")
            return collection
        else:
            client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []
    documents = []
    metadatas = []

    for doc in THYROID_LITERATURE:
        ids.append(doc["id"])
        documents.append(doc["text"])
        metadatas.append({
            "title": doc["title"],
            "source": doc["source"],
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    log.info(f"ChromaDB indexed: {collection.count()} documents")
    return collection
