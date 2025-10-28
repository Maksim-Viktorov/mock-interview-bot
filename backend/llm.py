import requests

def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "No feedback generated")
