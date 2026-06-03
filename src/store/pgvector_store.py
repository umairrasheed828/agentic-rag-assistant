import psycopg

from src.config import settings
from pgvector.psycopg import register_vector

EMBED_DIM = 768  # BAAI/bge-base-en-v1.5 produces 768-dim vectors


def init_db() -> None:
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id  TEXT PRIMARY KEY,
                arxiv_id  TEXT,
                title     TEXT,
                text      TEXT,
                embedding vector({EMBED_DIM})
            )
            """
        )
        conn.commit()
    print("Database initialized: 'chunks' table ready.")


if __name__ == "__main__":
    init_db()


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def insert_chunks(rows: list[dict]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, arxiv_id, title, text, embedding)
                VALUES (%(chunk_id)s, %(arxiv_id)s, %(title)s, %(text)s, %(embedding)s)
                ON CONFLICT (chunk_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding, text = EXCLUDED.text
                """,
                rows,
            )
        conn.commit()


def count_chunks() -> int:
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0


def fetch_all_chunks() -> list[dict]:
    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute("SELECT chunk_id, title, text FROM chunks").fetchall()
    return [{"chunk_id": r[0], "title": r[1], "text": r[2]} for r in rows]
