from ollama import AsyncClient
from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct

from settings import settings

embedding_model = SentenceTransformer(settings.embedding_model)
qdrant = AsyncQdrantClient(settings.qdrant_host, port=settings.qdrant_port)
ollama = AsyncClient(host=settings.ollama_host)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, qdrant

    if await qdrant.collection_exists(settings.collection_name):
        await qdrant.delete_collection(settings.collection_name)

    await qdrant.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    docs = [
        "FastAPI is a modern Python framework for building APIs.",
        "Liquibase is used to manage database migrations.",
        "Docker simplifies application deployment through containerization.",
    ]

    embeddings = embedding_model.encode(docs)
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"text": doc})
        for i, (doc, emb) in enumerate(zip(docs, embeddings))
    ]

    await qdrant.upsert(settings.collection_name, points)
    print(f"The ‘{settings.collection_name}’ collection has been created and populated.")

    yield

    await qdrant.delete_collection(settings.collection_name)
    await qdrant.close()


app = FastAPI(title="RAG Service", lifespan=lifespan)


@app.post("/ask")
async def ask(question: str = Body(..., embed=True)):
    query_vector = embedding_model.encode(question).tolist()

    search_result = await qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=1,
    )
    context = search_result.points[0].payload["text"]

    prompt = f"""
    Answer only based on the context. If you don't know, say “I don't know.”
    Context: {context}
    Question: {question}
    """

    responce = await ollama.chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )

    return {"answer": responce.message.content}
