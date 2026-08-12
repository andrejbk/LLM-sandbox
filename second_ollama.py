import ollama
from time import perf_counter


start_time = perf_counter()

response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Hi! Briefly explain what FastAPI is."}],
    options={"temperature": 0.2, "num_ctx": 2048},
)

response_time = perf_counter() - start_time

print(response["message"]["content"])
print(f"Response time: {response_time:.2f} sec")
