"""Document Q&A RAG app for Streamlit Community Cloud."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
MAX_FILE_BYTES = 10 * 1024 * 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
FALLBACK_ANSWER = "I could not find this in the document."


@dataclass
class DocumentIndex:
    """Text chunks and their normalized embedding vectors."""
    chunks: list[str]
    vectors: np.ndarray
    filename: str


@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per Streamlit server."""
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_text_bytes(filename: str, content: bytes) -> str:
    """Extract text from uploaded PDF or TXT bytes."""
    suffix = os.path.splitext(filename)[1].lower()
    if suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix == ".txt":
        text = content.decode("utf-8", errors="replace")
    else:
        raise ValueError("Please upload a PDF or TXT file.")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")
    return text


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_index(filename: str, content: bytes) -> DocumentIndex:
    """Extract, chunk and embed one uploaded document."""
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("File is larger than the 10 MB limit.")
    text = extract_text_bytes(filename, content)
    chunks = split_text(text)
    vectors = get_embedding_model().encode(chunks, normalize_embeddings=True)
    return DocumentIndex(chunks=chunks, vectors=np.asarray(vectors), filename=filename)


def retrieve(index: DocumentIndex, question: str, k: int = TOP_K) -> list[str]:
    """Find the top-k chunks by cosine similarity."""
    query_vector = get_embedding_model().encode([question], normalize_embeddings=True)[0]
    scores = index.vectors @ query_vector
    indices = np.argsort(scores)[::-1][:min(k, len(index.chunks))]
    return [index.chunks[int(i)] for i in indices]


def answer_question(index: DocumentIndex, question: str) -> str:
    """Answer using only passages retrieved from the uploaded document."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a question.")
    if len(question) > 500:
        raise ValueError("Question must be 500 characters or fewer.")
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        # Local runs may not have a Streamlit secrets file configured.
        api_key = ""
    api_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("Add ANTHROPIC_API_KEY in your Streamlit app secrets.")
    context = "\n\n---\n\n".join(retrieve(index, question))
    response = Anthropic(api_key=api_key).messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=700,
        temperature=0,
        system=(
            "Answer questions using only the supplied document excerpts. "
            f"If the answer is not clearly supported, reply exactly: {FALLBACK_ANSWER} "
            "Do not invent facts. Be concise."
        ),
        messages=[{"role": "user", "content": f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION:\n{question}"}],
    )
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts).strip() or FALLBACK_ANSWER


def main() -> None:
    """Render the Streamlit app."""
    st.set_page_config(page_title="Document Q&A RAG", page_icon="📄", layout="centered")
    st.title("📄 Document Q&A")
    st.write("Upload a PDF or TXT document, then ask questions answered from its contents.")
    st.caption("Demo app: do not upload sensitive or confidential documents.")

    uploaded = st.file_uploader("Choose a PDF or TXT file (maximum 10 MB)", type=["pdf", "txt"])
    if uploaded is not None:
        content = uploaded.getvalue()
        if len(content) > MAX_FILE_BYTES:
            st.error("File is larger than the 10 MB limit.")
            return
        signature = (uploaded.name, len(content), hash(content))
        if st.session_state.get("document_signature") != signature:
            with st.spinner("Extracting text and creating embeddings…"):
                try:
                    st.session_state["document_index"] = build_index(uploaded.name, content)
                    st.session_state["document_signature"] = signature
                    st.session_state["chat_history"] = []
                    st.success(f"Indexed {uploaded.name}: {len(st.session_state['document_index'].chunks)} chunks.")
                except Exception as exc:
                    st.session_state.pop("document_index", None)
                    st.error(f"Could not process document: {exc}")
                    return

    index = st.session_state.get("document_index")
    if index is None:
        st.info("Upload a document to begin.")
        return

    for item in st.session_state.get("chat_history", []):
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    question = st.chat_input("Ask a question about your document (max 500 characters)")
    if question:
        if len(question) > 500:
            st.error("Question must be 500 characters or fewer.")
            return
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Finding relevant passages and drafting an answer…"):
                try:
                    answer = answer_question(index, question)
                except Exception as exc:
                    answer = f"Error: {exc}"
                st.markdown(answer)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    if st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()


if __name__ == "__main__":
    main()
