"""
RAG-powered clinical Q&A for thyroid disease.
"""
import os
import logging
from rag.retriever import ThyroidRetriever

log = logging.getLogger(__name__)

LLM_PROVIDER = None
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        LLM_PROVIDER = "anthropic"
except ImportError:
    pass
if not LLM_PROVIDER:
    try:
        import openai
        if os.environ.get("OPENAI_API_KEY"):
            LLM_PROVIDER = "openai"
    except ImportError:
        pass


class ClinicalQA:
    def __init__(self):
        self.retriever = ThyroidRetriever()
        self.provider = LLM_PROVIDER
        self.client = None
        if self.provider == "anthropic":
            import anthropic as _a
            self.client = _a.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.provider == "openai":
            import openai as _o
            self.client = _o.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def ask(self, question, n_sources=3):
        if not question or not question.strip():
            return {"question": question, "answer": "Please enter a question.", "sources": []}

        sources = self.retriever.search(question, n_results=n_sources)
        context = self.retriever.get_context(question, n_results=n_sources)

        prompt = f"""You are a clinical knowledge assistant for thyroid disease.
Answer using ONLY the provided literature. Cite sources as [1], [2], etc.

QUESTION: {question}

LITERATURE:
{context}

Concise answer (3-5 sentences) with citations."""

        answer = None
        if self.provider and self.client:
            try:
                if self.provider == "anthropic":
                    r = self.client.messages.create(
                        model="claude-sonnet-4-20250514", max_tokens=300,
                        messages=[{"role": "user", "content": prompt}])
                    answer = r.content[0].text
                else:
                    r = self.client.chat.completions.create(
                        model="gpt-4o-mini", max_tokens=300,
                        messages=[{"role": "user", "content": prompt}])
                    answer = r.choices[0].message.content
            except Exception as e:
                log.warning(f"LLM failed: {e}")

        if not answer:
            answer = self._fallback(sources)

        return {
            "question": question,
            "answer": answer,
            "sources": [{"title": s["title"], "source": s["source"]} for s in sources],
        }

    def _fallback(self, sources):
        if not sources:
            return "No relevant medical literature found for this question."
        parts = []
        for i, s in enumerate(sources, 1):
            sentences = s["text"].split(". ")
            excerpt = ". ".join(sentences[:2]) + "."
            parts.append(f"**[{i}]** {excerpt}")
        return "Based on the indexed medical literature:\n\n" + "\n\n".join(parts)
