import requests
import json

OLLAMA_API = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.1:8b",
    "prompt": "Say hello in one sentence.",
    "stream": False
}

print("Testing Ollama connection...")
print(f"Sending request to: {OLLAMA_API}")

try:
    response = requests.post(OLLAMA_API, json=payload, timeout=60)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("SUCCESS!")
        print(f"Response: {result.get('response', 'No response field')}")
    else:
        print(f"ERROR: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to Ollama!")
    print("Make sure 'ollama serve' is running in another window.")
    
except requests.exceptions.Timeout:
    print("ERROR: Request timed out!")
    print("The model might be loading (this happens on first run).")
    print("Try running: ollama run llama3.1:8b")
    
except Exception as e:
    print(f"ERROR: {e}")