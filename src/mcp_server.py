from mcp.server.fastmcp import FastMCP

from src.store.pgvector_store import fetch_all_chunks

mcp = FastMCP("corpus-server")


@mcp.tool()
def corpus_overview() -> str:
    """Describe the knowledge base: how many papers it contains and sample titles."""
    titles = sorted({c["title"] for c in fetch_all_chunks()})
    sample = "\n".join(f"- {t}" for t in titles[:10])
    return f"The knowledge base contains {len(titles)} papers. Sample titles:\n{sample}"


if __name__ == "__main__":
    mcp.run()  # listens over stdio
