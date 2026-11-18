import os, requests
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

TESTS = [
    ("together",     "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("together",     "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    ("together",     "mistralai/Mistral-7B-Instruct-v0.1"),
    ("together",     "Qwen/Qwen2.5-72B-Instruct-Turbo"),
    ("fireworks-ai", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
    ("novita",       "meta-llama/llama-3.1-8b-instruct"),
    ("cohere",       "CohereForAI/c4ai-command-r-plus-08-2024"),
    ("sambanova",    "Meta-Llama-3.1-8B-Instruct"),
]

for provider, model in TESTS:
    url = f"https://router.huggingface.co/{provider}/v1/chat/completions"
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Bonjour"}], "max_tokens": 20},
            timeout=15,
        )
        icon = "✅" if r.status_code == 200 else "❌"
        print(f"{icon} [{r.status_code}] {provider} / {model}")
        if r.status_code == 200:
            print(f"   → {r.json()['choices'][0]['message']['content'][:80]}")
        else:
            print(f"   → {r.text[:80]}")
    except Exception as e:
        print(f"❌ {provider} / {model}: {e}")