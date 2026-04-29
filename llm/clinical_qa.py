"""
RAG-powered clinical Q&A for thyroid disease.

Usage:
    qa = ClinicalQA()
    answer = qa.ask("What are early signs of hypothyroidism?")
"""
import os
import logging
from typing import Optional
from rag.retriever import ThyroidRetriever

log = logging.getLogger(__name__)

# Try Anthropic first, then OpenAI
try:
    import anthropic
    LLM_PROVIDER = "anthropic"
except ImportError:
    try:
        import openai
        LLM_PROVIDER = "openai"
    except ImportError:
        LLM_PROVIDER = None


class ClinicalQA:
    """RAG-powered clinical question answering."""

    def __init__(self):
        self.retriever = ThyroidRetriever()
        self.provider = LLM_PROVIDER
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        elif self.provider == "openai":
            self.client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )

    def ask(self, question: str, n_sources: int = 3) -> dict:
        """
        Answer a clinical question using RAG.

        Args:
            question: Clinical question about thyroid disease
            n_sources: Number of literature sources to retrieve

        Returns:
            Dict with answer, sources, and context
        """
        sources = self.retriever.search(question, n_results=n_sources)
        context = self.retriever.get_context(question, n_results=n_sources)

        prompt = f"""You are a clinical knowledge assistant specializing in thyroid disease.
Answer the following question using ONLY the provided medical literature.
If the literature does not contain relevant information, say so.
Always cite your sources using [1], [2], etc.

QUESTION: {question}

MEDICAL LITERATURE:
{context}

Provide a concise, clinically accurate answer (3-5 sentences). Cite sources."""

        if self.provider == "anthropic":
            answer = self._call_anthropic(prompt)
        elif self.provider == "openai":
            answer = self._call_openai(prompt)
        else:
            answer = self._fallback(question, sources)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"title": s["title"], "source": s["source"]}
                for s in sources
            ],
        }

    def _call_anthropic(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"[LLM unavailable: {e}]"

    def _call_openai(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM unavailable: {e}]"

    def _fallback(self, question, sources):
        """Return retrieved context without LLM generation."""
        lines = [f"Retrieved {len(sources)} relevant sources:\n"]
        for s in sources:
            lines.append(f"- {s['title']} ({s['source']}): {s['text'][:200]}...")
        return "\n".join(lines)
