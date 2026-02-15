from typing import TypedDict, List, Optional
from backend.core.config import settings
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from backend.tools.query_expander import generate_hyde_document
from backend.tools.calculator import calculate
import json

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    original_question: str
    chat_history: str  
    documents: List[str]
    decision: str
    generation: str
    steps: List[str]
    retry_count: int
    mode: str
    temperature: float  
    max_tokens: int

# --- THE AGENT CLASS ---
class GraphRAGAgent:
    def __init__(self, retrieve_service, model_name=settings.OLLAMA_MODEL):
        self.retrieve_service = retrieve_service
        self.model_name = model_name 
        self.json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.max_retries = settings.MAX_ITERATIONS

    # --- NODES ---

    def router(self, state: AgentState):
        """Decides flow based on question."""
        print("---ROUTING QUESTION---")
        question = state.get("original_question", state["question"])
        
        # Simple routing prompt
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
        prompt = f"""
        Previous Chat History:
        {state.get('chat_history', '')}
        
        User: {state['original_question']}
        Reply politely and conversationally."""
        
        writer = ChatOllama(model=self.model_name, temperature=state["temperature"])
        res = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": res, "steps": ["general_conversation"]}

    def retrieve(self, state: AgentState):
        """
        Standard Vector Retrieval (No Neo4j Graph).
        """
        print(f"---RETRIEVING ({state['mode']})---")
        base_k = settings.TOP_K_RETRIEVAL
        top_k = int(base_k * 2) if state["mode"] == "detailed" else base_k
        try:
            search_query = state["question"]
            
            # Optional Upgrade: Add HyDE context if enabled
            if settings.USE_HYDE:
                hyde_context = generate_hyde_document(state["question"])
                search_query = state["question"] + "\n\n" + hyde_context
                
            # Perform standard search and extract text from metadata
            raw_docs = self.retrieve_service.retrieve(search_query, top_k=top_k)
            docs = [d.get("meta", {}).get("text", "") for d in raw_docs if "meta" in d]
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            docs = []
            
        return {"documents": docs, "steps": ["retrieve"]}

    def grade_documents(self, state: AgentState):
        print("---GRADING---")
        if not state["documents"]: return {"documents": []}
        
        doc_txt = "\n\n".join([f"[{i}] {d[:300]}..." for i, d in enumerate(state["documents"])])
        
        prompt = f"""Identify relevant docs for: {state['question']}
        Docs:
        {doc_txt}
        Return JSON {{ "indices": [0, 2...] }} of relevant docs containing ACTUAL content.
        If you are unsure, include the document."""
        
        try:
            res = self.json_llm.invoke([HumanMessage(content=prompt)])
            indices = json.loads(res.content).get("indices", [])
            filtered = [state["documents"][i] for i in indices if i < len(state["documents"])]
        except:
            filtered = state["documents"] 
        return {"documents": filtered, "steps": ["grade_documents"]}

    def transform_query(self, state: AgentState):
        print("---TRANSFORMING QUERY---")
        prompt = f"""
        Context: {state.get('chat_history', '')}
        User Question: {state['question']}
        
        Rewrite the user question to be standalone and search-friendly. Replace pronouns (he/she/it) with specific names from context if possible.
        Output ONLY the string."""
        
        new_q = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"question": new_q, "retry_count": state["retry_count"]+1}

    def generate(self, state: AgentState):
        print(f"---GENERATING ({state['mode'].upper()} | Tokens: {state['max_tokens']})---")
        question = state["original_question"]
        context = "\n\n".join(state["documents"])
        history = state.get("chat_history", "")
        token_limit = state.get("max_tokens", settings.MAX_TOKENS)        
        
        if state["mode"] == "detailed":
            system_prompt = (
                f"You are a comprehensive AI analyst. "
                f"Please provide a detailed, extensive answer using up to {token_limit} tokens if necessary. "
                "Cover all aspects of the context provided. Do NOT output raw JSON or mention your tools unless you are actively using them."
            )
        else:
            system_prompt = "You are a concise assistant. Answer directly and briefly. Do not output JSON."
        
        # REMOVED the hardcoded math prompt here, let Langchain's bind_tools handle the instruction natively.
        
        writer = ChatOllama(
            model=self.model_name, 
            temperature=state["temperature"],
            num_predict=token_limit
        )

        writer_with_tools = writer.bind_tools([calculate])
        prompt = f"""{system_prompt}
        
        Relevant Context:
        {context}
        
        Chat History:
        {history}
        
        Question: {question}
        Answer:"""
        
        messages = [HumanMessage(content=prompt)]
        
        # First pass checking if tool needs to be called
        response = writer_with_tools.invoke(messages)
        
        if response.tool_calls:
            messages.append(response) 
            
            for tool_call in response.tool_calls:
                if tool_call["name"] == "calculate":
                    math_expression = tool_call["args"].get("expression", "")
                    print(f"🛠️ LLM called Calculator for: {math_expression}")
                    
                    try:
                        math_result = calculate.invoke(tool_call["args"])
                        print(f"🧮 Calculator returned: {math_result}")
                        messages.append(ToolMessage(content=str(math_result), tool_call_id=tool_call["id"]))
                    except Exception as e:
                        print(f"🧮 Calculator failed: {e}")
                        messages.append(ToolMessage(content="Math calculation failed.", tool_call_id=tool_call["id"]))
            
            # Second pass after tool generates output
            response = writer_with_tools.invoke(messages)
        
        # Cleanup any hallucinated JSON that leaked into the final text
        res = response.content
        if "```json" in res and "calculate" in res:
            res = res.split("```json")[0].strip()
            
        return {"generation": res, "steps": ["generate"]}

    # --- EDGES & GRAPH ---
    def route_decision(self, state): return state["decision"]
    
    def decide_to_generate(self, state):
        if not state["documents"]:
            return "generate" if state["retry_count"] >= self.max_retries else "transform_query"
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
        workflow.add_conditional_edges("router", self.route_decision, {"chitchat":"chitchat", "vectorstore":"retrieve"})
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.decide_to_generate, {"transform_query":"transform_query", "generate":"generate"})
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("chitchat", END)
        workflow.add_edge("generate", END)
        return workflow.compile()

    def query(self, query: str, mode: str = "concise", temperature: float = 0.1, max_tokens: int = settings.MAX_TOKENS, chat_history: str = ""):
        app = self.build_graph()
        initial = {
            "question": query,
            "original_question": query,
            "chat_history": chat_history,
            "documents": [],
            "decision": "vectorstore",
            "retry_count": 0,
            "steps": [],
            "generation": "",
            "mode": mode,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        res = app.invoke(initial)
        return {"answer": res["generation"], "metadata": {"steps": res["steps"]}}