# frontend/chainlit_app.py
"""
Enhanced Chainlit demo UI for Agentic-RAG with improved answer presentation.

Requirements:
  pip install chainlit requests

Run:
  chainlit run frontend/chainlit_app.py --port 8001
"""

import os
import requests
import json
import chainlit as cl
from typing import Dict, List, Any

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_TOP_K = int(os.getenv("CHAINLIT_TOP_K", "5"))

def backend_query(query: str, top_k: int = DEFAULT_TOP_K, max_tokens: int = 512, temperature: float = 0.0):
    """
    Send a query to the FastAPI backend.
    """
    url = f"{BACKEND_URL}/query"
    payload = {
        "query": query, 
        "top_k": top_k, 
        "max_tokens": max_tokens, 
        "temperature": temperature
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"API Request failed: {e}")

def format_answer(answer: str) -> str:
    """
    Clean up and format the answer for better readability.
    """
    # Remove excessive newlines
    answer = '\n'.join(line for line in answer.split('\n') if line.strip())
    
    # Ensure proper spacing after periods
    answer = answer.replace('.', '. ').replace('.  ', '. ')
    
    return answer.strip()

def create_sources_display(sources: List[Dict[str, Any]]) -> str:
    """
    Create a rich, formatted display of retrieved sources.
    """
    if not sources:
        return ""
    
    md_lines = [f"### 📚 Retrieved Sources (Top {len(sources)})\n"]
    
    for i, s in enumerate(sources, 1):
        # Extract data with fallbacks
        idx = s.get("chunk_id") or s.get("index") or i
        score = s.get("score", 0)
        meta = s.get("metadata") or s.get("meta") or {}
        text_snip = s.get("text") or meta.get("text", "")
        
        # Color-code relevance score
        if score > 350:
            relevance = "🟢 High"
        elif score > 300:
            relevance = "🟡 Medium"
        else:
            relevance = "🔴 Low"
        
        # Create preview (first 300 chars)
        preview = text_snip[:300].replace("\n", " ").strip()
        if len(text_snip) > 300:
            preview += "..."
        
        # Build source card with preview only
        md_lines.append(f"#### Source {i} (Chunk {idx})")
        md_lines.append(f"**Relevance:** {relevance} (Score: {score:.2f})\n")
        md_lines.append(f"> {preview}\n")
        md_lines.append("---\n")
    
    return "\n".join(md_lines)

def extract_confidence_indicators(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the answer to provide confidence indicators.
    """
    indicators = {
        "has_citations": "chunk" in answer.lower() or "source" in answer.lower(),
        "avg_score": sum(s.get("score", 0) for s in sources) / len(sources) if sources else 0,
        "num_sources": len(sources),
        "answer_length": len(answer.split())
    }
    
    # Determine overall confidence
    if indicators["avg_score"] > 340 and indicators["num_sources"] >= 3:
        indicators["confidence"] = "High"
        indicators["emoji"] = "✅"
    elif indicators["avg_score"] > 300 and indicators["num_sources"] >= 2:
        indicators["confidence"] = "Medium"
        indicators["emoji"] = "⚠️"
    else:
        indicators["confidence"] = "Low"
        indicators["emoji"] = "❓"
    
    return indicators

@cl.on_chat_start
async def start():
    """
    Send a welcome message when the chat starts.
    """
    welcome_msg = f"""# 👋 Welcome to Agentic RAG Chat

I'm connected to the backend at `{BACKEND_URL}` and ready to answer questions about **The King of the Dark Chamber** and other ingested documents.

### 💡 Tips for best results:
- Ask specific questions about characters, themes, or plot points
- Request comparisons or analysis across multiple sections
- Ask for direct quotes or evidence from the text

**Try asking:** "Why does the King remain in the dark?" or "What is Queen Sudarshana's character arc?"
"""
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def handle_message(message: cl.Message):
    """
    Enhanced chat handler with improved answer presentation.
    """
    user_query = message.content.strip()
    if not user_query:
        await cl.Message(content="Please type a question.").send()
        return

    # Show thinking indicator
    thinking = await cl.Message(content="🔍 Searching through documents and generating answer...").send()

    try:
        # Call backend
        resp = await cl.make_async(backend_query)(
            user_query, top_k=DEFAULT_TOP_K, max_tokens=512, temperature=0.0
        )
    except Exception as e:
        thinking.content = f"❌ Backend call failed: {e}"
        await thinking.update()
        return

    # Extract response data
    raw_answer = resp.get("answer", "(no answer returned)")
    sources = resp.get("sources", [])
    
    # Format answer
    formatted_answer = format_answer(raw_answer)
    
    # Get confidence indicators
    confidence = extract_confidence_indicators(formatted_answer, sources)
    
    # Build enhanced answer display
    answer_display = f"""## {confidence['emoji']} Answer

{formatted_answer}

---
**Confidence:** {confidence['confidence']} | **Sources Used:** {confidence['num_sources']} | **Avg Relevance:** {confidence['avg_score']:.1f}
"""
    
    # Update thinking message with answer
    thinking.content = answer_display
    await thinking.update()

    # Display sources in separate message
    if sources:
        sources_display = create_sources_display(sources)
        await cl.Message(content=sources_display).send()
        
        # Send full text as Chainlit Elements for expandable view
        for i, s in enumerate(sources, 1):
            idx = s.get("chunk_id") or s.get("index") or i
            score = s.get("score", 0)
            text_snip = s.get("text") or s.get("metadata", {}).get("text", "")
            
            if text_snip:
                await cl.Text(
                    name=f"📄 Source {i} (Chunk {idx}) - Full Text",
                    content=text_snip,
                    display="side"
                ).send()
    else:
        await cl.Message(content="⚠️ No sources were retrieved for this query.").send()

    # Action buttons
    actions = [
        cl.Action(
            name="regen", 
            payload={"query": user_query, "temp": 0.7}, 
            label="🔄 Regenerate (creative)"
        ),
        cl.Action(
            name="expand", 
            payload={"query": user_query, "tokens": 1024}, 
            label="📝 Longer answer"
        ),
        cl.Action(
            name="json", 
            payload={"data": resp}, 
            label="🔧 Show raw data"
        )
    ]
    await cl.Message(content="### Actions:", actions=actions).send()

    # Store response for action callbacks
    cl.user_session.set("last_response", resp)
    cl.user_session.set("last_query", user_query)

@cl.action_callback("regen")
async def on_action_regen(action):
    """Regenerate with higher temperature for more creative answers."""
    query = action.payload.get("query")
    try:
        regen_msg = await cl.Message(content="🎲 Regenerating with more creativity...").send()
        regen_resp = await cl.make_async(backend_query)(
            query, top_k=DEFAULT_TOP_K, max_tokens=512, temperature=0.7
        )
        regen_answer = format_answer(regen_resp.get("answer", "(no answer)"))
        regen_msg.content = f"## 🎲 Regenerated Answer (Temperature: 0.7)\n\n{regen_answer}"
        await regen_msg.update()
    except Exception as e:
        await cl.Message(content=f"❌ Regeneration failed: {e}").send()

@cl.action_callback("expand")
async def on_action_expand(action):
    """Generate a longer, more detailed answer."""
    query = action.payload.get("query")
    try:
        expand_msg = await cl.Message(content="📝 Generating more detailed answer...").send()
        expand_resp = await cl.make_async(backend_query)(
            query, top_k=DEFAULT_TOP_K, max_tokens=1024, temperature=0.0
        )
        expand_answer = format_answer(expand_resp.get("answer", "(no answer)"))
        expand_msg.content = f"## 📝 Detailed Answer\n\n{expand_answer}"
        await expand_msg.update()
    except Exception as e:
        await cl.Message(content=f"❌ Expansion failed: {e}").send()

@cl.action_callback("json")
async def on_action_json(action):
    """Display raw JSON response for debugging."""
    resp = action.payload.get("data")
    json_str = json.dumps(resp, indent=2, ensure_ascii=False)
    await cl.Message(content=f"### 🔧 Raw JSON Response\n\n```json\n{json_str}\n```").send()