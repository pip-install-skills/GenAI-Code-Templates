import os, httpx, json

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ["OPENROUTER_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def chat_once(
    question="Say hi in one sentence.",
    model="openrouter/auto"  # or e.g. "openai/gpt-4o"
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
        return data

if __name__ == "__main__":
    print(chat_once())
