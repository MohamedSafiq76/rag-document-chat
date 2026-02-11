"""
RAG Chain Module
Handles LLM interaction with strict/hybrid mode prompt templates and source citations.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

client = InferenceClient(api_key=API_TOKEN)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

STRICT_SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to answer questions using the provided document context.

STRICT RULES:
1. Answer ONLY based on the information in the "Document Context" section below.
2. If the answer is NOT found in the context, respond with: "⚠️ I don't have enough information in the uploaded documents to answer this question."
3. NEVER use your general knowledge or make assumptions beyond what the documents state.
4. Always be specific and quote or reference the relevant parts of the documents.
5. When citing information, mention the source document name."""

HYBRID_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to uploaded documents. Your job is to provide helpful, accurate answers.

RULES:
1. Use the provided "Document Context" as your PRIMARY source of information.
2. If the document context contains relevant information, base your answer on it and cite the source.
3. If the document context is insufficient, you MAY supplement with your general knowledge, but clearly indicate when you are doing so by saying "Based on general knowledge:" or similar.
4. Always prioritize document-grounded answers over general knowledge.
5. When citing information from documents, mention the source document name."""


def _build_context_string(retrieved_docs):
    """Format retrieved document chunks into a readable context string."""
    if not retrieved_docs:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk", "")

        location = f"Source: {source}"
        if page:
            location += f", Page {page}"
        if chunk:
            location += f", Chunk {chunk}"

        context_parts.append(f"[{i}] {location}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


def _format_chat_history(chat_history, max_turns=5):
    """Format recent chat history into message list for the LLM."""
    messages = []
    # Take only the last N turns to stay within context limits
    recent = chat_history[-(max_turns * 2):]
    for role, content in recent:
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
    return messages


def _extract_sources(retrieved_docs):
    """Extract unique source citations from retrieved documents."""
    sources = []
    seen = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk", "")
        doc_type = doc.metadata.get("type", "")

        key = f"{source}-p{page}-c{chunk}"
        if key not in seen:
            seen.add(key)
            citation = {"source": source, "type": doc_type}
            if page:
                citation["page"] = page
            if chunk:
                citation["chunk"] = chunk
            # Include a snippet of the content for reference
            citation["snippet"] = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            sources.append(citation)
    return sources


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(question, retrieved_docs, chat_history=None, mode="strict"):
    """
    Generate an answer using the LLM with retrieved document context.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_docs : list[Document]
        Retrieved document chunks from the vector store.
    chat_history : list[tuple]
        List of (role, content) tuples for conversation context.
    mode : str
        Either "strict" (document-only) or "hybrid" (documents + general knowledge).

    Returns
    -------
    dict
        {"answer": str, "sources": list[dict]}
    """
    if chat_history is None:
        chat_history = []

    # Select prompt based on mode
    system_prompt = STRICT_SYSTEM_PROMPT if mode == "strict" else HYBRID_SYSTEM_PROMPT

    # Build context from retrieved docs
    context_string = _build_context_string(retrieved_docs)

    # Build the user message with context
    user_message = f"""Document Context:
{context_string}

Question: {question}

Please provide a detailed answer based on the above context."""

    # Construct messages list
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history for conversational context
    history_messages = _format_chat_history(chat_history)
    messages.extend(history_messages)

    # Add current question with context
    messages.append({"role": "user", "content": user_message})

    # Call LLM
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1024,
            temperature=0.3 if mode == "strict" else 0.5,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"❌ Error generating response: {str(e)}"

    # Extract source citations
    sources = _extract_sources(retrieved_docs)

    return {"answer": answer, "sources": sources}
