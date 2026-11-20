"""
This compatibility script intentionally does NOT build or rebuild vectors.
Dami's vectors already live in Supabase. Run this only as a safe connection
check after setting the environment variables.
"""
from src.config import validate_settings
from src.embeddings import embed_query
from src.vector_store import match_documents


def main():
    validate_settings()
    print("Creating a temporary query embedding...")
    vector = embed_query("آزمایش اتصال پایگاه دانش دامی")
    print(f"Embedding dimension: {len(vector)}")
    print("Calling existing match_dami_documents RPC...")
    rows = match_documents(vector, match_count=1, filter_metadata={})
    print(f"RPC success. Returned rows: {len(rows)}")
    if rows:
        print("First result fields:", ", ".join(rows[0].keys()))
        print("Similarity:", rows[0].get("similarity"))


if __name__ == "__main__":
    main()
