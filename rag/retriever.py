"""
Semantic retrieval over thyroid medical literature.

Usage:
    retriever = ThyroidRetriever()
    results = retriever.search("What causes elevated TSH?")
"""
import logging
from typing import List, Dict
from rag.indexer import build_index

log = logging.getLogger(__name__)


class ThyroidRetriever:
    """Semantic search over indexed thyroid medical literature."""

    def __init__(self):
        self.collection = build_index()

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant medical literature."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return documents

    def get_context(self, query: str, n_results: int = 3) -> str:
        """Get formatted context string for LLM prompt."""
        docs = self.search(query, n_results)
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[{i}] {doc['title']} ({doc['source']})\n{doc['text']}"
            )
        return "\n\n".join(context_parts)
