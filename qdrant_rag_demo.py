from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct

documents = [
    "FastAPI is a modern web framework for building APIs in Python.",
    "Alembic is used to manage database migrations in SQLAlchemy.",
    "Docker simplifies the deployment of applications through containerization.",
]

model = SentenceTransformer("all-MiniLM-L6-V2")
embeddings = model.encode(documents)

client = QdrantClient("localhost", port=6333)

collection_name = "docs"
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=embedding.tolist(),
        payload={"text": doc, "source": "internal_docs"}
    )
    for i, (doc, embedding) in enumerate(zip(documents, embeddings))
]

client.upload_points(collection_name=collection_name, points=points)

query = "How do I update the database structure?"
query_vector = model.encode(query).tolist()

search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1,
)

print(search_results.points[0].payload["text"])
