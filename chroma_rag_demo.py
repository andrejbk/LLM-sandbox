import chromadb
from chromadb.utils import embedding_functions

text = """
FastAPI is a modern web framework for building APIs in Python.
It automatically generates documentation and supports asynchronous operations.

Alembic is a tool for managing database migrations in SQLAlchemy.
It allows you to safely update the database schema when models change.

Docker is a platform for containerizing applications.
It simplifies deployment and isolates dependencies.
"""

chunks = text.strip().split("\n\n")
print("We broke the text down into chunks:", len(chunks))

print("Let's configure the embedding function...")
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

print("Save in Chroma...")
client = chromadb.EphemeralClient()
collection = client.create_collection(name="docs", embedding_function=emb_func)
collection.add(
    ids=[f"doc_{i}" for i in range(len(chunks))],
    documents=chunks
)

query = "How do I update the database structure?"
print(f"\nQuery: {query}")
results = collection.query(
    query_texts=[query],
    n_results=1
)

retrieved = results["documents"][0][0]
print(f"\nA relevant excerpt has been found: {retrieved}")
print("\nThis excerpt can be provided to an LLM as context for generating"
      " a response")
