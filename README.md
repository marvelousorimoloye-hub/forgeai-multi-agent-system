# ForgeAI 

**Multi-Agent Adaptive Research & Synthesis Engine**

An advanced multi-agent system that transforms complex research queries into high-quality, citation-rich reports using **hierarchical context engineering**, **hybrid RAG**, and **agentic reflection**.

Built as a portfolio showcase for production-grade AI engineering.

---

## ✨ Key Features

- **Hierarchical Context Engineering** — Multi-level (L1 Summary + L2 Findings + L3 Evidence) context orchestration
- **Hybrid Retrieval** — Vector + BM25 with Voyage AI reranking (Cohere fallback)
- **Smart Merge & Deduplication** — Prevents context poisoning from web + local sources
- **Agentic Reflection Loop** — Self-correcting Critic Agent
- **Multi-Turn Conversation** — Vector memory + LLM summarization for long sessions
- **Full Observability** — Ready for LangSmith / Phoenix

---

## Tech Stack

| Layer                | Technology                              |
|----------------------|-----------------------------------------|
| Orchestration        | LangGraph                               |
| Retrieval            | Chroma + Hybrid (Vector + BM25)         |
| Reranking            | Voyage AI (`rerank-2-lite`) + Cohere fallback |
| Embeddings           | BAAI/bge-small-en-v1.5                  |
| LLMs                 | Groq + Gemini 2.5                       |
| Memory               | Vector Memory + LLM Summarization       |
| Evaluation           | RAGAS + Custom Critic                   |

---

## Project Structure


src/forgeai/
├── agents/                 # Supervisor, Retrievers, Reranker, Critic, etc.
├── rag/                    # Vector store + hybrid retrieval
├── graphs/                 # Main LangGraph workflow
├── config/                 # Prompts & Settings
├── utils/                  # LLM factory, Pydantic models
└── monitoring/             # Future dashboard


## Quick start
```bash
# 1. Clone & install
git clone https://github.com/yourusername/forgeai.git
cd forgeai
pip install -e .

# 2. Set up environment variables
cp .env.example .env
# Add your keys: GROQ, GOOGLE (Gemini), TAVILY, VOYAGE, COHERE

# 3. Ingest documents
python -c "from src.forgeai.rag.vector_store import auto_ingest_on_startup; auto_ingest_on_startup()"

# 4. Run interactive mode
python -m src.forgeai.main
```

## Roadmap

 Streamlit Monitoring Dashboard
 Automated Benchmark Suite
 Multi-modal RAG (images + text)
 Human-in-the-loop feedback


Built with ❤️ in Lagos, Nigeria
## This project demonstrates:

Advanced Context Engineering
Production-grade RAG pipelines
Agentic systems with reflection loops
Scalable memory management
