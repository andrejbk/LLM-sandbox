import ollama
from time import perf_counter


start_time = perf_counter()

response = ollama.generate(
    model="mistral",
    prompt="Hi! Briefly explain what FastAPI is.",
)

response_time = perf_counter() - start_time

print(response["response"])
print(f"Response time: {response_time:.2f} sec")
