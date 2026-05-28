import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\nCRITICAL ERROR: OPENAI_API_KEY not found in environment variables!")
else:
    print(f"\nDEBUG: OPENAI_API_KEY loaded. Starts with: {api_key[:7]}...")

model = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=api_key,
    temperature=0.7,
    max_tokens=4096,
)
