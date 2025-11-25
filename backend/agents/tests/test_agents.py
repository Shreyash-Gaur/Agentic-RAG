# backend/agents/tests/test_agents.py
import sys, os
sys.path.append(os.path.abspath("backend"))

from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent
from agents.rag_agent import RAGAgent

class DummyRetriever:
    def __init__(self):
        pass
    def retrieve(self, q, top_k=5):
        return [{"index": 1, "score": 0.1, "meta": {"text": "dummy text 1"}},
                {"index": 2, "score": 0.2, "meta": {"text": "dummy text 2"}}]

class DummyOllama:
    def generate(self, model, prompt, max_tokens=512, temperature=0.0):
        return "generated answer"

def test_researcher():
    r = ResearcherAgent(DummyRetriever())
    out = r.research("hello", top_k=2)
    assert isinstance(out, list)
    assert out[0]["index"] == 1

def test_writer():
    w = WriterAgent(ollama_client=DummyOllama(), model="mymodel")
    res = w.generate_answer("q", [{"index":1,"meta":{"text":"t"}}])
    assert res["success"] is True
    assert "answer" in res and res["answer"] == "generated answer"

def test_rag_agent():
    r = ResearcherAgent(DummyRetriever())
    w = WriterAgent(ollama_client=DummyOllama(), model="mymodel")
    rag = RAGAgent(r, w, memory=None)
    out = rag.query("who?", top_k=2)
    assert "answer" in out
