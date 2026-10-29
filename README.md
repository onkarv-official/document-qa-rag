# Document Q&A — RAG (Streamlit)

A beginner-friendly Retrieval-Augmented Generation demo. Upload a PDF or TXT file, retrieve semantically relevant passages, and ask Anthropic to answer using those passages.

## Features
- PDF and TXT files, maximum 10 MB
- Character chunks of 800 with 100-character overlap
- Embeddings from `sentence-transformers/all-MiniLM-L6-v2`
- Top 4 passages retrieved for each question
- Anthropic-generated answers grounded in retrieved excerpts
- Fallback: `I could not find this in the document.`
- API key supplied through hosting secrets, not source code

## Deploy from your phone

1. Create a new GitHub repository named `document-qa-rag` (public is simplest for Streamlit Community Cloud).
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.env.example`, `DEPLOYMENT.md`, and `tests/test_app_helpers.py`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your `document-qa-rag` repository, branch `main`, and main file `app.py`.
5. In advanced deployment settings, select Python 3.11 if a Python version selector is shown.
6. In the app's **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "your_real_anthropic_api_key"
   ```
7. Deploy and test with a non-sensitive PDF or TXT.

## Secrets
Required:
- `ANTHROPIC_API_KEY` — create/get this in the Anthropic Console. Keep it private.

Optional environment variables:
- `ANTHROPIC_MODEL` — default `claude-3-5-haiku-latest`
- `EMBEDDING_MODEL` — default `sentence-transformers/all-MiniLM-L6-v2`

Never commit real secrets, put them in screenshots, or send them in chat. Anthropic API usage may cost money independently of hosting.

## Local setup
Python 3.11 recommended.
```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run app.py
pytest -q
```
Set `ANTHROPIC_API_KEY` as an environment variable or in Streamlit secrets before asking questions.

## Limitations
- Scanned PDFs need OCR before upload.
- Model download/startup may be slow on free hosting.
- In-memory uploaded document indexes are not durable and may be cleared on restart.
- Do not upload sensitive documents. Prompting helps ground answers but cannot fully prevent prompt injection in document content.
- Verify answers against the source document.
