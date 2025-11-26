# frontend/chainlit_app.py
"""
Chainlit demo UI for Agentic-RAG.

Requirements:
  pip install chainlit requests

Run:
  # run your backend (uvicorn backend.main:APP --host 127.0.0.1 --port 8000)
  # run chainlit on a different port (8001)
  chainlit run frontend/chainlit_app.py --port 8001

The UI will call the backend /query endpoint at BACKEND_URL (default http://localhost:8000).
"""

import os
import requests
import json
import chainlit as cl

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_TOP_K = int(os.getenv("CHAINLIT_TOP_K", "5"))

def backend_query(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = 512, temperature: float = 0.0):
    url = f"{BACKEND_URL}/query"
    payload = {"query": query, "top_k": top_k, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

@cl.on_chat_start
async def start():
    # Greeting message when a new chat starts
    await cl.Message(
        content=(
            f"Hello — this Chainlit UI talks to the Agentic-RAG backend at {BACKEND_URL}.\n\n"
            "Ask any question about the ingested documents (the King of the Dark Chamber book is loaded)."
        )
    ).send()

@cl.on_message
async def handle_message(message: str, conversation: cl.Conversation):
    """
    Called for each user message in Chainlit.
    - Calls backend /query
    - Shows the answer, prompt, and retrieved sources
    - Adds a 'Regenerate' button (runs same query with higher temperature)
    """
    user_query = message.strip()
    if not user_query:
        await conversation.send_message("Please type a question.")
        return

    # show a typing / thinking message
    thinking = await cl.Message(content="Retrieving and generating…").send()

    try:
        resp = backend_query(user_query, top_k=DEFAULT_TOP_K, max_tokens=512, temperature=0.0)
    except Exception as e:
        await thinking.update(content=f"Backend call failed: {e}")
        return

    answer = resp.get("answer", "(no answer returned)")
    prompt = resp.get("prompt", "")
    sources = resp.get("sources", [])

    # Update thinking message with the answer
    await thinking.update(content=f"**Answer:**\n\n{answer}")

    # Display sources as a separate message with clickable toggles
    if sources:
        md_lines = ["### Retrieved sources (top {})".format(len(sources))]
        for s in sources:
            idx = s.get("index")
            score = s.get("score")
            meta = s.get("meta", {})
            text_snip = meta.get("text", "")
            # show first 350 chars as preview
            preview = text_snip[:350].replace("\n", " ").strip()
            md_lines.append(f"**Chunk {idx}**  \nScore: `{score:.3f}`  \n{preview}...  \n")
            # add a hidden block with full text (Chainlit renders markdown so we can include collapsible details)
            md_lines.append(f"<details><summary>Full chunk {idx}</summary>\n\n```\n{text_snip}\n```\n</details>\n")

        sources_md = "\n\n".join(md_lines)
        await cl.Message(content=sources_md).send()
    else:
        await cl.Message(content="(No retrieved sources)").send()

    # Show the assembled prompt in a code block so user can inspect or copy
    if prompt:
        await cl.Message(content="**Assembled prompt sent to the generator (for debugging):**\n\n```text\n" + prompt + "\n```").send()

    # Add quick action buttons: Regenerate with temp 0.7 and show raw JSON
    async def on_regen():
        try:
            regen_resp = backend_query(user_query, top_k=DEFAULT_TOP_K, max_tokens=512, temperature=0.7)
            regen_answer = regen_resp.get("answer", "(no answer)")
            await cl.Message(content="**Regenerated (temperature=0.7):**\n\n" + regen_answer).send()
        except Exception as e:
            await cl.Message(content=f"Regeneration failed: {e}").send()

    async def on_show_json():
        await cl.Message(content="```json\n" + json.dumps(resp, indent=2, ensure_ascii=False) + "\n```").send()

    # Buttons in Chainlit are created via UI elements:
    await cl.Button(name="Regenerate (temp=0.7)", on_click=on_regen).send()
    await cl.Button(name="Show raw JSON", on_click=on_show_json).send()
