from flashrank import Ranker, RerankRequest

ranker = Ranker()

query = "How do I roll back a migration in Alembic?"

passages = [
    {"id": 0, "text": "Alembic is a tool for managing database migrations."},
    {"id": 1, "text": "The `alembic downgrade` command allows you to roll back a migration."},
    {"id": 2, "text": "Migrations are applied using `alembic upgrade head`."},
]

request = RerankRequest(query=query, passages=passages)

results = ranker.rerank(request)

print(results)
