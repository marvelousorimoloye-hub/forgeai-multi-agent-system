import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from pathlib import Path
from src.forgeai.config.settings import settings


# Force offline mode + use local cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Use the already downloaded model from cache
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.path.expanduser("~/.cache/huggingface/hub"),   # Your existing cache
    model_kwargs={
        "device": "cpu",
        "token": settings.hf_token if settings.hf_token else None,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)

print(f"✅ Using cached embedding model: {embeddings.model_name}")

def get_vector_store():
    persist_dir = Path(settings.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="forgeai_knowledge"
    )

def auto_ingest_on_startup(directory: str = "data/seed_papers"):
    """Robust ingestion with progress and batching"""
    vector_store = get_vector_store()
    
    if len(vector_store.get()['ids']) > 0:
        print(f"✅ Vector store already has {len(vector_store.get()['ids'])} documents.")
        return

    print("📥 Starting document ingestion...")

    try:
        loader = PyPDFDirectoryLoader(
            directory,
            glob="**/*.pdf",
            silent_errors=True,
            extract_images=False
        )
        documents = loader.load()
        
        if not documents:
            print("⚠️ No PDFs found or readable.")
            return

        print(f"Found {len(documents)} pages. Starting embedding in batches...")

        # === Added: Consistent source_type for all ingested documents ===
        for doc in documents:
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source_type"] = "local"
            doc.metadata["source"] = "local"   # kept for compatibility

        # Add documents in smaller batches to avoid hanging
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
            print(f"  → Embedded batch {i//batch_size + 1}/{(len(documents)+batch_size-1)//batch_size} ({len(batch)} pages)")

        print(f"✅ Successfully ingested {len(documents)} pages into Chroma.")

    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        import traceback
        traceback.print_exc()

        
def get_hybrid_retriever(k: int = 8):
    """Return Hybrid Retriever (Vector + BM25) with Reranking"""
    vector_store = get_vector_store()
    
    # Vector Retriever
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": k * 2}
    )
    
    # BM25 Retriever (Keyword-based)
    try:
        # Get all documents for BM25
        docs = vector_store.get()
        if len(docs['documents']) > 0:
            bm25_retriever = BM25Retriever.from_texts(
                texts=docs['documents'],
                metadatas=[{"source": "local"} for _ in docs['documents']]
            )
            bm25_retriever.k = k * 2
        else:
            bm25_retriever = None
    except:
        bm25_retriever = None

    # Ensemble (Hybrid) Retriever
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]   # Give more weight to semantic search
        )
        return ensemble_retriever
    else:
        return vector_retriever