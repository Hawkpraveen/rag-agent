# RAG agent

Python 3.11+ FastAPI service for document ingestion, semantic retrieval, and
cited answers using local Ollama models. Entry point: `main.py`, instance: `app`.
This service runs on port 3000 and is called by the Node backend, not React.

```powershell
cd D:\Projects\RAG\rag-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3000 --reload
```

The environment is already installed locally. Keep Ollama running. API docs:
http://127.0.0.1:3000/docs. Open the React UI at http://localhost:8080 for normal use.

`POST /ingest` accepts `{"path":"docs"}`. `POST /ask` and `POST /search` accept
`{"question":"What is the refund policy?","top_k":4,"min_score":0.25}`.
Relative document and database paths resolve against this agent directory.
`GET /health` reports service liveness; it does not check model readiness.

Optional environment variables: `OLLAMA_URL` (http://127.0.0.1:11434),
`EMBED_MODEL` (nomic-embed-text), `CHAT_MODEL` (llama3.2:3b), and `RAG_DB`
(data/index.sqlite3). Pull models before selecting them. Re-ingest all documents
or use a new database when changing the embedding model.

The CLI is also available:

```powershell
.\.venv\Scripts\python.exe rag.py ingest docs
.\.venv\Scripts\python.exe rag.py ask "What is the refund policy?"
.\.venv\Scripts\python.exe rag.py search "support hours"
.\.venv\Scripts\python.exe rag.py chat
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`rag.py` contains ingestion, retrieval, SQLite storage, and the Ollama client.
`docs/` contains a sample handbook; `tests/` uses a deterministic fake model.
The ignored `data/` directory stores document text and embeddings.

Text and Markdown are chunked with character offsets; PDFs retain page numbers.
Scanned PDFs need OCR first. Unchanged files are skipped, and changed files are
replaced only after embeddings succeed. Deleted files remain indexed until the
index is rebuilt. Vector retrieval scans the index in Python and suits small
collections. Similarity is not confidence; calibrate the threshold for your data.

Each question uses one retrieval pass and an answer call (Auto can retry once
after answer validation fails), with no conversation
memory or autonomous tools. Citations identify retrieved excerpts; they do not
prove that an answer is correct. Review sources for important claims. Documents
are untrusted input, and prompt instructions cannot fully prevent injection.

Keep this unauthenticated service bound to localhost. Local document text is
sent to your configured Ollama endpoint; a remote endpoint changes where data goes.


File attachments: choose the paperclip Attach button in the chat or Attach a file in the library. Supported formats: .txt, .md, and text-based .pdf; maximum 10 MB per file. Files travel through Node to the agent and are retained under rag-agent/data/uploads for citations. Node then publishes a MongoDB dataset snapshot. Scanned PDFs require OCR. Start the agent with .venv\Scripts\python.exe run.py so HOST and PORT in rag-agent/.env are used. Node reads PORT and AGENT_URL from rag-BE/.env. Restart Node after changing its configuration.

## Chat usage reporting

POST /ask also returns token_usage entries for each embedding/chat call made for
that question. Counts come from Ollama prompt_eval_count and eval_count; missing
counts remain null. Error responses retain usage already reported. Node owns
MongoDB conversation and usage persistence. Indexing/upload/search usage is outside
this chat-persistence change.

## Model routing

`POST /ask` accepts `model_mode`: `auto` (default), `fast`, or `thorough`.
Auto uses question intent and supplied context size/document count. Routine
lookups and short summaries use `FAST_CHAT_MODEL` (defaults to `CHAT_MODEL`).
Comparisons, detailed analysis, questions over 600 characters, context over 6500
characters, or more than three documents use `THOROUGH_CHAT_MODEL` (default
`qwen3.5:4b`). Code questions use `CODE_CHAT_MODEL` (default `qwen2.5-coder:3b`).
These are heuristics, not a calibrated measure of difficulty or answer quality.

The agent checks installed models through Ollama `/api/tags` on each question.
Unavailable preferred models fall back to an installed configured fast/default
model, with the reason included in the response. Nothing is downloaded.
Explicit modes override task routing. Only Auto retries validation failures once
on a different thorough model; network errors do not trigger another generation.
The `routing` response records mode, selected model, reason, and initial model
when retried. Every completed model call retains its own token usage.
Embedding/index settings are independent of chat model selection.
Restart the agent after changing these server environment settings.
