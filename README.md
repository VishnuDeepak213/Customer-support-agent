# Agentforce-inspired Customer Support Agent

This repository implements an Agentforce-inspired support agent using LangChain, Groq LLaMA 3.1, FastAPI, and Gradio.

Quickstart (local):

1. Create the database:

```bash
python scripts/generate_crm.py
```

2. Set your Groq API key in environment:

```bash
export GROQ_API_KEY=your_api_key_here
```

3. Run the FastAPI app:

```bash
uvicorn app.main:app --reload --port 8000
```

4. Run the Gradio UI:

```bash
python gradio_app.py
```

5. Run evaluation (50 queries):

```bash
python scripts/evaluate.py
```

Docker (quick):

```bash
docker build -t agentforce-demo .
docker run -e GROQ_API_KEY=$GROQ_API_KEY -p 8000:8000 -p 7860:7860 agentforce-demo
```

Notes:
- This variant removes Phase 4 (Guardrails) as requested. The agent still uses ReAct tools and logs conversations and escalations in SQLite.
- Replace the Groq LLM integration with another LLM if you do not have a Groq key.
