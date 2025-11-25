# frontend/chainlit_app.py
import chainlit as cl
import requests
import os

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

@cl.on_chat_start
async def start():
    await cl.Message(content="Agentic-RAG Chainlit UI ready. Ask me anything!").send()

@cl.on_message
async def main(message: str):
    # simple pipeline: call /chat to preserve memory and get RAG answer
    payload = {"conv_id": "demo", "query": message, "top_k": 5}
    try:
        r = requests.post(f"{BACKEND}/chat", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = data.get("answer") or "No answer"
        sources = data.get("sources", [])
        await cl.Message(content=text).send()
        if sources:
            await cl.Message(content="Sources:").send()
            for s in sources:
                meta = s.get("meta",{})
                snippet = meta.get("text","")[:300].replace("\n"," ")
                await cl.Message(content=f"- idx:{s.get('index')} score:{s.get('score')}\n{snippet}").send()
    except Exception as e:
        await cl.Message(content=f"Error calling backend: {e}").send()
