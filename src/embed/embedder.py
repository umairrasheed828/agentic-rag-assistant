import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed documents/chunks for storage. No prefix needed for bge-v1.5 passages."""
    return np.asarray(get_model().encode(texts, normalize_embeddings=True))


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. bge-v1.5 recommends this instruction prefix for retrieval."""
    prefixed = "Represent this sentence for searching relevant passages: " + text
    return np.asarray(get_model().encode([prefixed], normalize_embeddings=True)[0])
