from mistralai.client import Mistral, errors
from time import perf_counter
import os
import sys
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["MISTRAL_API_KEY"]
MODEL = "mistral-small-latest"
PROMPT = "Hi! Briefly explain what FastAPI is."

client = Mistral(api_key=API_KEY)

start_time = perf_counter()

try:
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "user",
             "content": PROMPT,
             },
        ],
        temperature=0.2,
        max_tokens=100,
    )
except errors.mistralerror as e:
    print(f"Mistral API error: {e}")
    sys.exit(1)

response_time = perf_counter() - start_time

print(response.choices[0].message.content)
print(f"Response time: {response_time:.2f} sec")
