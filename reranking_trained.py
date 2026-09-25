from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("./reranker-trained", local_files_only=True,)

query = "How do I roll back a migration in Alembic?"

passages = [
    {"id": 0, "text": "Alembic is a tool for managing database migrations."},
    {"id": 1, "text": "The `alembic downgrade` command allows you to roll back a migration."},
    {"id": 2, "text": "Migrations are applied using `alembic upgrade head`."},
]

query_embedding = model.encode(query, convert_to_tensor=True)

passage_embeddings = model.encode(
    [passage["text"] for passage in passages],
    convert_to_tensor=True,
)

scores = util.cos_sim(query_embedding, passage_embeddings)[0].tolist()

results = [
    {**passage, "score": score}
    for passage, score in zip(passages, scores)
]

results.sort(key=lambda item: item["score"], reverse=True)

for result in results:
    print(
        f"id={result['id']} | "
        f"score={result['score']:.4f} | "
        f"{result['text']}"
    )
