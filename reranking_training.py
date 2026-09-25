from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

train_dataset = Dataset.from_list([
    {
        "anchor": "How do I roll back a migration?",
        "positive": "alembic downgrade - reverts the last migration",
        "negative": "alembic upgrade - applies migrations",
    },
    {
        "anchor": "Creating a user via the API",
        "positive": 'POST /api/v1/users {"name": "..."}',
        "negative": "DELETE /api/v1/users/{id} deletes the user",
    },
])

model = SentenceTransformer("all-MiniLM-L6-v2")

loss = MultipleNegativesRankingLoss(model)
trainer = SentenceTransformerTrainer(
    model=model,
    args=SentenceTransformerTrainingArguments(
        output_dir="./reranker-trained",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=0,
    ),
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()
model.save_pretrained("./reranker-trained")
