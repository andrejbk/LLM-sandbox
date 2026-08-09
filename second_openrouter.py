from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from time import perf_counter
import os
import sys


load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-oss-20b:free"
PROMPT = "Hi! Briefly explain what FastAPI is."

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

start_time = perf_counter()

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user",
             "content": PROMPT,
             },
        ],
        temperature=0.2,
        max_tokens=300,
    )
except OpenAIError as e:
    print(f"OpenRouter API error: {e}")
    sys.exit(1)

response_time = perf_counter() - start_time

print(response.choices[0].message.content)
print(f"Response time: {response_time:.2f} sec")
