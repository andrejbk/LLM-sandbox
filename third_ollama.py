import ollama
from time import perf_counter


start_time = perf_counter()

for chunk in ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Hi! Briefly explain what FastAPI is."}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)

response_time = perf_counter() - start_time

print()
print(f"Response time: {response_time:.2f} sec")
