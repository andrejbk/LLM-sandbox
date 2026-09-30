import ollama
import chromadb
from chromadb.utils import embedding_functions
from sympy.physics.units import temperature

docs = [
    "FastAPI is a modern Python framework for building APIs.",
    "Liquibase is used to manage database migrations.",
    "Docker simplifies application deployment through containerization.",
]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

collection = client.create_collection(
    name="docs",
    embedding_function=ef
)

collection.add(
    ids=[f"id_{i}" for i in range(len(docs))],
    documents=docs
)


def ask_rag(question: str) -> str:
    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    context = results["documents"][0][0]

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content":
                "Answer only based on the following context. "
                "If you do not know the answer, say 'I don't know.'"},
            {"role": "user", "content":
             f"Context: {context}\n\nQuestion:{question}"},
        ],
        options={
            "temperature": 0.1,
        },
    )

    return response.message.content


if __name__ == "__main__":

    question = "Which migration tool is mentioned in the documentation?"
    answer = ask_rag(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
