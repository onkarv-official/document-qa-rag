# Deployment checklist

- [ ] Create a new repository named `document-qa-rag` in your GitHub account.
- [ ] Upload all files from this folder to the repository root (keep `tests/` as a folder).
- [ ] Visit https://share.streamlit.io/ and sign in with GitHub.
- [ ] Create app from repository, branch `main`, entry file `app.py`.
- [ ] Choose Python 3.11 in advanced settings if the version selector is available.
- [ ] In Streamlit app Settings → Secrets, add `ANTHROPIC_API_KEY = "your_real_key"` in TOML format.
- [ ] Wait for the build and test with a small, non-sensitive PDF/TXT.
- [ ] Ask a question answered in the document and a question not answered in the document.

Required secret: `ANTHROPIC_API_KEY`.
Optional variables: `ANTHROPIC_MODEL`, `EMBEDDING_MODEL`.

Never commit `.env`, API keys, uploaded files, or confidential documents.
