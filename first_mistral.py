from mistralai.client import Mistral
from time import perf_counter
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

start_time = perf_counter()
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "user",
         "content": "Hi! Briefly explain what FastAPI is.",
         },
    ],
    temperature=0.2,
    max_tokens=100,
)
response_time = perf_counter() - start_time

print(response.choices[0].message.content)
print(f"Response time: {response_time:.2f} sec")
