import hashlib
from typing import Dict, Any
from langchain_core.messages import AIMessage

from src.forgeai.utils.pydantic_models import ResearchState


async def merge_documents_node(state: ResearchState) -> Dict[str, Any]:
    """
    Merge documents from Web + Knowledge Retriever
    - Smart deduplication using hashlib
    - Adds source_type to prevent context poisoning
    - Keeps the best version of duplicate content
    """
    print("→ Entering Merge Documents Node")

    raw_docs = state.get("raw_documents", [])
    if not raw_docs:
        return {
            "raw_documents": [],
            "messages": [AIMessage(content="Merge: No documents to merge.")]
        }

    try:
        seen_hashes = {}
        unique_docs = []
        WEB_DEFAULT_SCORE = 0.65

        for doc in raw_docs:
            content = doc.get("page_content", "").strip()
            if not content:
                continue

            # Create hash for deduplication
            content_norm = content.lower()[:800]
            content_hash = hashlib.md5(content_norm.encode()).hexdigest()

            # === Improved source_type detection (clean & consistent) ===
            metadata = doc.get("metadata", {}).copy()
            source_type = metadata.get("source_type") or metadata.get("source") or \
                         ("web" if metadata.get("url") else "local")
            metadata["source_type"] = source_type

            current_score = doc.get("relevance_score", WEB_DEFAULT_SCORE if source_type == "web" else 0.5)

            # Deduplication + keep better version
            if content_hash in seen_hashes:
                existing_idx = seen_hashes[content_hash]
                if current_score > unique_docs[existing_idx].get("relevance_score", 0):
                    unique_docs[existing_idx] = {
                        "page_content": content,
                        "metadata": metadata,
                        "relevance_score": current_score
                    }
                continue

            # Add new unique document
            seen_hashes[content_hash] = len(unique_docs)
            unique_docs.append({
                "page_content": content,
                "metadata": metadata,
                "relevance_score": current_score
            })

        print(f"✅ Merge completed: {len(raw_docs)} → {len(unique_docs)} unique documents")

        return {
            "raw_documents": unique_docs,
            "messages": [AIMessage(content=f"Merge: {len(unique_docs)} unique documents after deduplication.")],
        }

    except Exception as e:
        print(f"❌ Merge error: {e}")
        return {
            "raw_documents": raw_docs[:15],  # fallback
            "messages": [AIMessage(content=f"Merge failed, using original documents. Error: {str(e)}")],
        }