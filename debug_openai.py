from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")
print(f"Key: {os.getenv('OPENAI_API_KEY')[:5]}...")

client = OpenAI()

try:
    print("Testing connection...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("Response received:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
