def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split text into chunks of `chunk_size` words, with optional overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
