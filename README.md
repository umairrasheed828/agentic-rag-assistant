# Agentic RAG Assistant

![CI](https://github.com/umairrasheed828/agentic-rag-assistant/actions/workflows/ci.yml/badge.svg)

An agentic retrieval-augmented generation system over a corpus of AI-research papers —
built to be **rigorously evaluated and calibrated**, then containerised and auto-deployed to AWS.

Ask a question; a LangGraph agent decides when to retrieve, runs **hybrid search with
reranking** over ~50 arXiv papers, and answers grounded in the sources. Answer quality is
measured by an **LLM-as-judge that is calibrated against human labels** — so the quality
numbers can actually be trusted.

## Architecture

    question
       |
       v
    LangGraph agent  --(tool: search_papers)-->  Hybrid retrieval:
       |                                           dense (bge-base + pgvector)
       |                                           + sparse (BM25)  --RRF-->
       |                                           cross-encoder rerank -> top 5
       v
    grounded answer   (served via FastAPI /ask; Streamlit demo UI)

## Highlights

- **Hybrid retrieval** — dense embeddings (BAAI/bge-base-en-v1.5 in Postgres/pgvector)
  fused with sparse BM25 via Reciprocal Rank Fusion, then **cross-encoder reranking**
  (bge-reranker-base).
- **Agentic** — a LangGraph agent with tools (`search_papers`, `corpus_overview`),
  also exposed over **MCP** (server + client) with input and recursion guardrails.
- **Rigorously evaluated** — retrieval **Recall@5 = 1.00, MRR = 0.833**; an LLM-as-judge
  scores faithfulness / relevance / correctness, **calibrated against human labels**
  (judge MAE **0.60 -> 0.40** after rubric tightening; relevance MAE 0.00).
- **Production-ready** — FastAPI service (`/health`, `/ask`), Dockerised with
  Postgres/pgvector via docker-compose, deployed on **AWS EC2 with GitHub Actions CI/CD**,
  plus a Streamlit demo UI and LangSmith tracing.

## Quickstart

    uv sync
    # create a .env with OPENAI_API_KEY (+ optional LangSmith vars)

    docker compose up -d                         # Postgres + pgvector
    uv run python -m src.ingest.download         # fetch arXiv papers
    uv run python -m src.ingest.chunk_corpus     # chunk
    uv run python -m src.store.pgvector_store    # create the table
    uv run python -m src.ingest.index_chunks     # embed + index

    uv run uvicorn src.api:app --port 8000       # serve -> http://127.0.0.1:8000/docs
    uv run streamlit run streamlit_app.py        # optional UI -> http://localhost:8501

Query from the CLI:

    uv run python -m src.agent.agent "How do you evaluate large language models?"

## Evaluation (the focus)

Retrieval is measured on a gold set: **Recall@5 = 1.00**, **MRR = 0.833**.

Answer quality uses an LLM-as-judge (faithfulness, relevance, correctness). Because an
unaudited judge is just another unverified component, the judge is **calibrated against
human labels**: tightening the rubric cut faithfulness/correctness error from
**MAE 0.60 to 0.40 with no regressions**, reproducing the known result that LLM judges drift
lenient on subtle cases. Reproduce via `eval/run_retrieval_eval.py`, `eval/llm_judge.py`,
and `eval/calibration.py`.

## Tech

Python 3.12 · LangChain / LangGraph · OpenAI gpt-4o-mini · Postgres / pgvector ·
BAAI/bge-base-en-v1.5 + bge-reranker-base · rank-bm25 · MCP · FastAPI · Docker ·
AWS EC2 · GitHub Actions CI/CD · LangSmith · Streamlit. Env/deps via uv; gates: ruff, mypy, pytest.

## Status

Complete: RAG core with hybrid retrieval + reranking, agentic orchestration
(LangGraph + tools + MCP), retrieval & answer evaluation with judge calibration, and a
Dockerised FastAPI service auto-deployed to AWS via CI/CD.