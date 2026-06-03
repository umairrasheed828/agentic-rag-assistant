from sentence_transformers import CrossEncoder

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
