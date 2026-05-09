import requests
import os

API = os.getenv("API_URL", "http://127.0.0.1:8000")

def ask(q):
    r = requests.post(f"{API}/chat", json={"query": q, "session_id": "demo"})
    print(r.status_code)
    print(r.json())

if __name__ == '__main__':
    ask("I can't login and my subscription payment failed — what's my account status? My email is test@example.com")
