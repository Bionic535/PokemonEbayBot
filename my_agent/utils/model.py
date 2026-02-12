import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("\nCRITICAL ERROR: NVIDIA_API_KEY not found in environment variables!")
else:
    print(f"\nDEBUG: NVIDIA_API_KEY loaded. Starts with: {api_key[:4]}...")


# Initialize model explicitly using NVIDIA
model = ChatNVIDIA(
    model="nvidia/nemotron-3-nano-30b-a3b",
    api_key=api_key,
    temperature=1,
    top_p=1,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)