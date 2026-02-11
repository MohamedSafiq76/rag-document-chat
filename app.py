"""
RAG Chat Application — Streamlit UI
Chat with your enterprise documents using Retrieval-Augmented Generation.
"""

import streamlit as st
from ingest import ingest_files, SUPPORTED_EXTENSIONS
from vector_store import add_documents, search, get_document_count, clear
from rag_chain import generate_answer

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="📚 RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* Mode badge styling */
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .mode-strict {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .mode-hybrid {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #93c5fd;
    }

    /* Source citation cards */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .source-card strong {
        color: #334155;
    }

    /* Stats in sidebar */
    .stat-box {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-box h3 {
        margin: 0;
        color: #0284c7;
        font-size: 1.5rem;
    }
    .stat-box p {
        margin: 0;
        color: #64748b;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "strict"
if "file_names" not in st.session_state:
    st.session_state.file_names = []

# ---------------------------------------------------------------------------
# Sidebar — Configuration & Document Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Mode Toggle
    st.markdown("### 🎯 Response Mode")
    mode_choice = st.radio(
        "Select how the AI should respond:",
        ["🔒 Strict Mode (Document-Only)", "🔄 Hybrid Mode (Documents + Knowledge)"],
        index=0 if st.session_state.mode == "strict" else 1,
        help="Strict = answers only from documents. Hybrid = documents first, general knowledge as fallback.",
    )
    st.session_state.mode = "strict" if "Strict" in mode_choice else "hybrid"

    if st.session_state.mode == "strict":
        st.markdown('<div class="mode-badge mode-strict">🔒 STRICT — Document-Only Answers</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-badge mode-hybrid">🔄 HYBRID — Documents + General Knowledge</div>', unsafe_allow_html=True)

    st.divider()

    # Document Upload
    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or CSV files",
        type=["pdf", "docx", "csv"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, CSV",
    )

    # Process Documents Button
    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("📄 Ingesting and embedding documents..."):
                try:
                    # Ingest files into chunks
                    documents = ingest_files(uploaded_files)

                    if not documents:
                        st.error("❌ No text content could be extracted from the uploaded files.")
                    else:
                        # Store in Chroma
                        add_documents(documents)

                        st.session_state.documents_loaded = True
                        st.session_state.file_names = [f.name for f in uploaded_files]

                        st.success(f"✅ Processed {len(uploaded_files)} file(s) → {len(documents)} chunks embedded!")
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")

    st.divider()

    # Knowledge Base Stats
    st.markdown("### 📊 Knowledge Base")
    doc_count = get_document_count()
    st.markdown(f"""
    <div class="stat-box">
        <h3>{doc_count}</h3>
        <p>Chunks in Vector DB</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.file_names:
        st.markdown("**Loaded Documents:**")
        for name in st.session_state.file_names:
            st.markdown(f"- 📄 {name}")

    st.divider()

    # Clear Knowledge Base
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        clear()
        st.session_state.chat_history = []
        st.session_state.documents_loaded = False
        st.session_state.file_names = []
        st.success("Knowledge base cleared!")
        st.rerun()

# ---------------------------------------------------------------------------
# Main Area — Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>📚 RAG Document Chat</h1>
    <p>Upload enterprise documents and ask questions — get grounded, cited answers</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main Area — Chat Interface
# ---------------------------------------------------------------------------

# Display chat history
for entry in st.session_state.chat_history:
    role, content = entry["role"], entry["content"]
    with st.chat_message(role):
        st.markdown(content)
        # Show sources for assistant messages
        if role == "assistant" and "sources" in entry and entry["sources"]:
            with st.expander(f"📎 Sources ({len(entry['sources'])} references)", expanded=False):
                for src in entry["sources"]:
                    source_info = f"📄 **{src['source']}**"
                    if "page" in src:
                        source_info += f" — Page {src['page']}"
                    if "chunk" in src:
                        source_info += f", Chunk {src['chunk']}"
                    snippet = src.get('snippet', '')
                    st.info(f"{source_info}\n\n_{snippet}_")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents are loaded
    if not st.session_state.documents_loaded and get_document_count() == 0:
        st.warning("⚠️ Please upload and process documents first using the sidebar.")
    else:
        # Mark documents as loaded if DB has content (from a previous session)
        if not st.session_state.documents_loaded and get_document_count() > 0:
            st.session_state.documents_loaded = True

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating answer..."):
                # Semantic search
                retrieved_docs = search(prompt, k=5)

                # Build conversation history for the LLM
                history_for_llm = [
                    (entry["role"], entry["content"])
                    for entry in st.session_state.chat_history
                ]

                # Generate answer
                result = generate_answer(
                    question=prompt,
                    retrieved_docs=retrieved_docs,
                    chat_history=history_for_llm,
                    mode=st.session_state.mode,
                )

                # Display answer
                st.markdown(result["answer"])

                # Display sources
                if result["sources"]:
                    with st.expander(f"📎 Sources ({len(result['sources'])} references)", expanded=False):
                        for src in result["sources"]:
                            source_info = f"📄 **{src['source']}**"
                            if "page" in src:
                                source_info += f" — Page {src['page']}"
                            if "chunk" in src:
                                source_info += f", Chunk {src['chunk']}"
                            snippet = src.get('snippet', '')
                            st.info(f"{source_info}\n\n_{snippet}_")

        # Save to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

# ---------------------------------------------------------------------------
# Empty State — Getting Started Guide
# ---------------------------------------------------------------------------
if not st.session_state.chat_history and not st.session_state.documents_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 1️⃣ Upload Documents
        Upload PDF, DOCX, or CSV files using the sidebar. These will be your knowledge base.
        """)

    with col2:
        st.markdown("""
        ### 2️⃣ Choose Mode
        **Strict Mode** 🔒 — Answers only from documents (zero hallucination).
        **Hybrid Mode** 🔄 — Documents first, general knowledge as backup.
        """)

    with col3:
        st.markdown("""
        ### 3️⃣ Start Chatting
        Ask questions and get grounded answers with source citations!
        """)
