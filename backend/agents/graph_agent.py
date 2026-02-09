from typing import TypedDict, List, Optional
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import json

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    original_question: str
    documents: List[str]
    decision: str
    generation: str
    steps: List[str]
    retry_count: int
    mode: str
    temperature: float  
    session_id: str 

# --- THE AGENT CLASS ---
class GraphRAGAgent:
    def __init__(self, retrieve_service, memory_service=None, model_name="mistral"):
        self.retrieve_service = retrieve_service
        self.memory_service = memory_service 
        self.model_name = model_name 
        
        # Router and Grader need strict logic (Low Temp)
        self.json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self.llm = ChatOllama(model=model_name, temperature=0)
        
        self.max_retries = 3

    # --- NODES ---

    def router(self, state: AgentState):
        """Decides flow based on question."""
        print("---ROUTING QUESTION---")
        question = state.get("original_question", state["question"])
        prompt = f"""You are a router. 
        1. If user asks for info/facts/summary, output 'vectorstore'.
        2. If user says hi/hello/thanks, output 'chitchat'.
        Question: {question}
        Return JSON: {{ "datasource": "vectorstore" | "chitchat" }}"""
        
        try:
            res = self.json_llm.invoke([HumanMessage(content=prompt)])
            decision = json.loads(res.content).get("datasource", "vectorstore")
        except:
            decision = "vectorstore"
        return {"decision": decision, "steps": ["router"]}

    def general_conversation(self, state: AgentState):
        # Fetch history for chitchat
        history_str = ""
        if self.memory_service and state.get("session_id"):
            hist = self.memory_service.get_history(state["session_id"])
            if hist:
                history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in hist[-4:]])

        prompt = f"""Conversation History:
        {history_str}
        
        User: {state['original_question']}
        Reply politely."""

        writer = ChatOllama(model=self.model_name, temperature=state["temperature"])
        res = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": res, "steps": ["general_conversation"]}

    def retrieve(self, state: AgentState):
        print(f"---RETRIEVING ({state['mode']})---")
        # Use the transformed question for retrieval
        q_to_search = state["question"]
        print(f"   (Searching for: '{q_to_search}')")
        
        top_k = 8 if state["mode"] == "detailed" else 5
        try:
            results = self.retrieve_service.retrieve(q_to_search, top_k=top_k)
            docs = [r['meta'].get('text', '') for r in results]
        except:
            docs = []
        return {"documents": docs, "steps": ["retrieve"]}

    def grade_documents(self, state: AgentState):
        print("---GRADING---")
        if not state["documents"]: return {"documents": []}
        
        doc_txt = "\n\n".join([f"[{i}] {d}" for i, d in enumerate(state["documents"])])
        prompt = f"""Identify relevant docs for: {state['question']}
        Docs:
        {doc_txt}
        Return JSON {{ "indices": [0, 2...] }} of relevant docs containing ACTUAL content."""
        
        try:
            res = self.json_llm.invoke([HumanMessage(content=prompt)])
            indices = json.loads(res.content).get("indices", [])
            filtered = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except:
            filtered = state["documents"] 
        return {"documents": filtered, "steps": ["grade_documents"]}

    def transform_query(self, state: AgentState):
        print("---TRANSFORMING QUERY---")
        
        # 1. Fetch History
        history_str = ""
        if self.memory_service and state.get("session_id"):
            hist = self.memory_service.get_history(state["session_id"])
            if hist:
                # Use last 6 turns for context
                history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in hist[-6:]])

        # 2. Rewrite Question
        prompt = f"""Given the conversation history, rewrite the user's latest question to be a standalone search query.
        Resolve any pronouns (he, she, it, they) to their actual names from the history.
        
        History:
        {history_str}
        
        User's latest question: {state['original_question']}
        
        Output ONLY the rewritten query string (no quotes)."""
        
        new_q = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print(f"   (Rewritten: '{new_q}')")
        
        return {"question": new_q, "retry_count": state["retry_count"]+1}

    def generate(self, state: AgentState):
        print(f"---GENERATING ({state['mode'].upper()} | Temp: {state['temperature']})---")
        question = state["original_question"]
        context = "\n\n".join(state["documents"])
        
        # Memory Injection for Generation
        history_block = ""
        if self.memory_service and state.get("session_id"):
            hist = self.memory_service.get_history(state["session_id"])
            if hist:
                history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in hist])
                history_block = f"Previous Conversation:\n{history_str}\n"

        if state["mode"] == "detailed":
            system_prompt = "You are a comprehensive analyst. Write a detailed, in-depth response. Minimum 300 words."
        else:
            system_prompt = "You are a concise assistant. Answer directly and briefly."

        writer = ChatOllama(model=self.model_name, temperature=state["temperature"])

        prompt = f"""{system_prompt}
        
        You are a faithful assistant. You MUST answer the question using ONLY the information provided in the Context below.
        If the Context does not contain the answer, say "I cannot find the answer in the document."
        DO NOT use your own outside knowledge.

        {history_block}
        
        Context: 
        {context}
        
        Current Question: {question}
        Answer:"""
        
        res = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": res, "steps": ["generate"]}

    # --- EDGES & GRAPH ---
    def route_decision(self, state): return state["decision"]
    def decide_to_generate(self, state):
        if not state["documents"]:
            # If no docs found, TRY REWRITING THE QUERY FIRST
            if state["retry_count"] < self.max_retries:
                return "transform_query"
            return "generate"
        return "generate"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router)
        workflow.add_node("chitchat", self.general_conversation)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("generate", self.generate)

        workflow.set_entry_point("router")
        
        # Logic: Router -> Vectorstore -> Transform Query (first time) -> Retrieve
        # We force a transform first to handle "he/she" questions immediately?
        # Alternatively, we can let Router decide. 
        # For now, let's stick to standard flow: Router -> Retrieve. 
        # If Retrieve fails (empty docs), it goes to Transform.
        # BUT for "how old is he", retrieval WON'T fail (it finds Stan). 
        # IMPROVEMENT: We should Always transform if it's a follow-up. 
        # For simplicity, let's keep the existing flow but rely on the user to click "transform" logic 
        # OR we can add a Conditional Edge after router to check if history exists.
        
        workflow.add_conditional_edges("router", self.route_decision, {"chitchat":"chitchat", "vectorstore":"retrieve"})
        workflow.add_edge("retrieve", "grade_documents")
        
        # If grading fails, we transform.
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {"transform_query":"transform_query", "generate":"generate"})
        
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("chitchat", END)
        workflow.add_edge("generate", END)
        return workflow.compile()

    def query(self, query: str, mode: str = "concise", temperature: float = 0.1):
        app = self.build_graph()
        session_id = "default_session"

        # 1. Write User Turn
        if self.memory_service:
            self.memory_service.add_turn(session_id, "user", query)

        initial = {
            "question": query,
            "original_question": query,
            "documents": [],
            "decision": "vectorstore",
            "retry_count": 0,
            "steps": [],
            "generation": "",
            "mode": mode,
            "temperature": temperature,
            "session_id": session_id
        }
        res = app.invoke(initial)
        
        # 2. Write Assistant Turn
        if self.memory_service:
            self.memory_service.add_turn(session_id, "assistant", res["generation"])

        return {"answer": res["generation"], "metadata": {"steps": res["steps"]}}