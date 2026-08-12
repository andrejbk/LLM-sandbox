import requests
from time import perf_counter

from httpx import stream

start_time = perf_counter()

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": "Hi! Briefly explain what FastAPI is.",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 2048,
        },
    },
)

response_time = perf_counter() - start_time

print(response.json()["response"])
print(f"Response time: {response_time:.2f} sec")
