# save this as test_ollama.py
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:latest",
        "prompt": "Reply with one word only: hello",
        "stream": False
    }
)

print(response.json()["response"])
