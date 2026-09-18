# FastAgentAPI RAG Agent

A retrieval-augmented, tool-using conversational agent built on LangGraph, exposed through a FastAPI service. It combines dense vector search (ChromaDB) with sparse keyword search (BM25) and fuses the two using Reciprocal Rank Fusion — the same principle a search engine uses when it blends "meaning match" with "exact word match" results rather than trusting either alone.

## Architecture

```
START → retrieval (Hybrid RAG) → LLM (tool-bound) ⇄ tools → END
```

1. **Retrieval node** — rewrites the incoming query for search, then pulls candidates from ChromaDB (semantic) and BM25 (keyword), merges them via RRF, and injects the top 5 chunks as context.
2. **LLM node** — a Groq-hosted model (`openai/gpt-oss-120b`) bound to four tools, decides whether to answer directly or call a tool.
3. **Tool node** — executes the selected tool and routes back to the LLM node until the model produces a final answer.
4. **Memory** — `MemorySaver` checkpoints state per `thread_id`, giving each conversation persistent short-term memory.

## Tools

| Tool | Purpose |
|---|---|
| `calculator` | Evaluates arithmetic expressions via `numexpr` |
| `weather` | Fetches current conditions from `wttr.in` |
| `web_search` | Falls back to DuckDuckGo when retrieved context is insufficient |
| `file_handler` | Sandboxed create/read/append/delete on a local `agent_files/` directory |

## Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM:** Groq (`ChatGroq`)
- **Vector store:** ChromaDB (persistent, default embedding function)
- **Keyword search:** `rank_bm25` (Okapi)
- **API layer:** FastAPI, `slowapi` (rate limiting), SQLAlchemy (`QueryHistory` model)
- **Ingestion:** `PyPDFLoader` + `RecursiveCharacterTextSplitter`

## Setup

1. Install dependencies:
   ```bash
   pip install langchain langchain-community langchain-groq langchain-text-splitters \
               langgraph chromadb rank_bm25 numexpr duckduckgo-search \
               fastapi slowapi sqlalchemy python-dotenv requests
   ```
2. Create a `.env` file:
   ```
   API_KEY=your_internal_api_key
   GROQ_API_KEY=your_groq_key
   ```
3. Place the source PDF (`Why_Language_Models_Hallucinate_Explainer.pdf`) in the project root, or update the loader path.
4. Provide a `database.py` exposing `QueryHistory` (SQLAlchemy model) and `create_db` (session dependency) — referenced but not defined in the current script.
5. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Ask a question; runs the full LangGraph pipeline (rate-limited, 5/min) |
| `GET` | `/ask` | Retrieve stored question/answer history |
| `PUT` | `/ask` | Update a stored question by `id` |
| `DELETE` | `/ask` | Delete a stored record by `id` |

All endpoints require an `x-api-key` header matching `API_KEY`.

## Project Structure

```
.
├── app.py                # Main script (graph + FastAPI app)
├── database.py            # SQLAlchemy models + session (required, not included above)
├── VectorDB/               # Chroma persistent store (auto-created)
├── agent_files/            # Sandbox for file_handler tool (auto-created)
└── .env
```

## Known Limitations

- The BM25 index is built once at startup over the single ingested PDF — re-ingesting new documents requires a restart.
- `/chat` builds its own `config` from `request.thread_id`, which is correct, but the module-level `config` (`thread_id: user_1`) used during the standalone test call at the bottom of the script will collide with that thread if left in the deployed file — strip that test invocation before shipping.
- No streaming; each `/chat` call blocks until the full graph run completes.
- `file_handler` is sandboxed to `agent_files/` via `os.path.basename`, but there's no per-user isolation — all API consumers share the same file space.