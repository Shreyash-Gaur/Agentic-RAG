# backend/agents/graph_agent.py
"""
Agentic RAG — upgraded graph agent.

What changed from the original:
  1. Graph is compiled ONCE in __init__ (not rebuilt on every request).
  2. Planner node: LLM decomposes the question into tool steps at runtime.
  3. execute_step node: runs one plan step (retrieve / calculate / summarize).
     Multiple retrieve steps → different sub-queries → richer context.
  4. reflect node: LLM critiques its own answer and decides what to do next.
  5. targeted_retrieve node: reflection-driven extra retrieval when answer
     is incomplete, without rebuilding the whole plan.
  6. document_sources collected and returned so callers can see evidence.
  7. All bare `except:` replaced with `except Exception:`.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from backend.core.config import settings
from backend.tools.calculator import calculate
from backend.tools.query_expander import generate_hyde_document

logger = logging.getLogger("agentic-rag.agent")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── core ──────────────────────────────────────────────────────────────
    question: str                   # possibly rewritten
    original_question: str
    chat_history: str
    mode: str                       # "concise" | "detailed"
    temperature: float
    max_tokens: int

    # ── retrieval ─────────────────────────────────────────────────────────
    documents: List[str]            # accumulated text chunks
    document_sources: List[Dict]    # metadata for each chunk (for API sources)

    # ── routing ───────────────────────────────────────────────────────────
    decision: str                   # "vectorstore" | "chitchat"
    generation: str

    # ── agentic planning ──────────────────────────────────────────────────
    plan: List[Dict]                # [{tool, input, reason}, ...]
    plan_step_idx: int              # which step we are currently executing
    tool_results: List[str]         # free-text results from each step

    # ── reflection ────────────────────────────────────────────────────────
    reflection_verdict: str         # "good" | "needs_more" | "replan"
    reflection_query: str           # refined query suggested by reflection
    reflection_count: int           # guard against infinite loops

    # ── misc ──────────────────────────────────────────────────────────────
    retry_count: int
    steps: List[str]                # node names visited (for debugging)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GraphRAGAgent:
    """
    LangGraph-based agentic RAG agent.

    The graph is compiled once at construction time and reused for every
    request — fixing the previous bug where build_graph() was called on
    every query() invocation.
    """

    MAX_PLAN_STEPS = 4          # cap to keep local models tractable
    MAX_REFLECTIONS = 2         # max reflection→retry cycles

    def __init__(self, retrieve_service, model_name: str = settings.OLLAMA_MODEL):
        self.retrieve_service = retrieve_service
        self.model_name = model_name

        # Two LLM variants — reuse across requests
        self._json_llm = ChatOllama(model=model_name, temperature=0, format="json")
        self._llm      = ChatOllama(model=model_name, temperature=0)

        # Compile graph ONCE
        self._app = self._build_graph()
        logger.info("GraphRAGAgent compiled and ready (model=%s)", model_name)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _writer(self, temperature: float, max_tokens: int) -> ChatOllama:
        """Create a generation-time LLM with the caller's params."""
        return ChatOllama(
            model=self.model_name,
            temperature=temperature,
            num_predict=max_tokens,
        )

    def _invoke_json(self, prompt: str, fallback: Dict) -> Dict:
        """Call the JSON LLM and parse the result; return fallback on error."""
        try:
            res = self._json_llm.invoke([HumanMessage(content=prompt)])
            return json.loads(res.content)
        except Exception as e:
            logger.warning("JSON LLM parse failed: %s", e)
            return fallback

    def _retrieve_docs(self, query: str, top_k: int) -> tuple[List[str], List[Dict]]:
        """
        Run HyDE-augmented retrieval.

        FIX vs original: we embed only the hypothetical document (not
        query + hyde concatenated), which is what the HyDE paper actually
        prescribes. The original query is used for display / grading only.
        """
        search_query = query
        if settings.USE_HYDE:
            try:
                hyde_doc = generate_hyde_document(query)
                # Correct HyDE: embed the HYPOTHETICAL DOC alone
                search_query = hyde_doc
            except Exception as e:
                logger.warning("HyDE generation failed, using raw query: %s", e)

        raw = self.retrieve_service.retrieve(search_query, top_k=top_k)
        texts   = [r.get("meta", {}).get("text", "") for r in raw if "meta" in r]
        sources = [r.get("meta", {})                  for r in raw if "meta" in r]
        return texts, sources

    # -----------------------------------------------------------------------
    # Graph nodes
    # -----------------------------------------------------------------------

    def _router(self, state: AgentState) -> Dict:
        """Decide between chitchat and the agentic RAG path."""
        logger.info("--- ROUTER ---")
        question = state.get("original_question", state["question"])
        prompt = (
            f"You are a router.\n"
            f"1. If the user wants information, facts, or a summary → 'vectorstore'.\n"
            f"2. If the user says hi / thanks / small talk → 'chitchat'.\n"
            f"Question: {question}\n"
            f"Return JSON: {{\"datasource\": \"vectorstore\" | \"chitchat\"}}"
        )
        result = self._invoke_json(prompt, {"datasource": "vectorstore"})
        decision = result.get("datasource", "vectorstore")
        return {"decision": decision, "steps": ["router"]}

    def _chitchat(self, state: AgentState) -> Dict:
        """Handle small talk without retrieval."""
        logger.info("--- CHITCHAT ---")
        prompt = (
            f"Previous chat:\n{state.get('chat_history', '')}\n\n"
            f"User: {state['original_question']}\n"
            f"Reply politely and conversationally."
        )
        writer = self._writer(state["temperature"], state["max_tokens"])
        reply  = writer.invoke([HumanMessage(content=prompt)]).content
        return {"generation": reply, "steps": ["chitchat"]}

    # ── Planner ─────────────────────────────────────────────────────────────

    def _planner(self, state: AgentState) -> Dict:
        """
        NEW: LLM decomposes the question into a sequence of tool steps.

        Available tools:
          - retrieve  : search the knowledge base for a sub-topic
          - calculate : evaluate a mathematical expression
          - summarize : ask the LLM to condense already-retrieved text

        The planner is the key difference between a pipeline and an agent.
        A pipeline decides the steps at design time; the planner decides
        them at runtime based on the question.
        """
        logger.info("--- PLANNER ---")
        prompt = f"""You are a research planner for a RAG system.
Given the user's question, produce a step-by-step plan using only these tools:
  - retrieve  : search the document knowledge base (provide a focused sub-query)
  - calculate : evaluate a math expression (provide the expression string)
  - summarize : condense retrieved context (provide a short instruction)

Rules:
  - Use between 1 and {self.MAX_PLAN_STEPS} steps.
  - Prefer multiple focused retrieve steps over one broad retrieve.
  - Only include calculate if the question involves arithmetic.
  - Only include summarize if the question asks for a summary/overview.

Chat history:
{state.get('chat_history', 'None')}

Question: {state['original_question']}

Return ONLY valid JSON:
{{
  "steps": [
    {{"tool": "retrieve", "input": "<focused sub-query>", "reason": "<why>"}},
    ...
  ]
}}"""

        result   = self._invoke_json(prompt, {"steps": []})
        raw_plan = result.get("steps", [])

        # Sanitize: keep only known tools, cap to MAX_PLAN_STEPS
        valid_tools = {"retrieve", "calculate", "summarize"}
        plan = [
            s for s in raw_plan
            if isinstance(s, dict) and s.get("tool") in valid_tools
        ][:self.MAX_PLAN_STEPS]

        # Fallback: at least one retrieve step
        if not plan:
            logger.warning("Planner produced empty plan — falling back to single retrieve.")
            plan = [{"tool": "retrieve", "input": state["original_question"], "reason": "fallback"}]

        logger.info("Plan (%d steps): %s", len(plan), [s["tool"] for s in plan])
        return {
            "plan": plan,
            "plan_step_idx": 0,
            "tool_results": [],
            "document_sources": [],
            "steps": ["planner"],
        }

    # ── Step executor ────────────────────────────────────────────────────────

    def _execute_step(self, state: AgentState) -> Dict:
        """
        NEW: Execute one step from the plan.

        After each step, increments plan_step_idx so the graph can decide
        whether to loop back here or move on to generate.
        """
        idx  = state["plan_step_idx"]
        plan = state["plan"]
        step = plan[idx]
        tool = step.get("tool", "retrieve")
        inp  = step.get("input", state["question"])

        logger.info("--- EXECUTE STEP %d/%d  tool=%s ---", idx + 1, len(plan), tool)

        new_docs    = list(state.get("documents", []))
        new_sources = list(state.get("document_sources", []))
        tool_results = list(state.get("tool_results", []))

        if tool == "retrieve":
            top_k = settings.TOP_K_RETRIEVAL * 2 if state["mode"] == "detailed" else settings.TOP_K_RETRIEVAL
            texts, sources = self._retrieve_docs(inp, top_k)
            new_docs.extend(texts)
            new_sources.extend(sources)
            summary = f"[Retrieve: '{inp}'] → {len(texts)} chunks found."
            tool_results.append(summary)
            logger.info("Retrieved %d chunks for sub-query: '%s'", len(texts), inp)

        elif tool == "calculate":
            try:
                result = calculate.invoke({"expression": inp})
                tool_results.append(f"[Calculate: {inp}] → {result}")
            except Exception as e:
                tool_results.append(f"[Calculate: {inp}] → Error: {e}")

        elif tool == "summarize":
            context = "\n\n".join(new_docs[-4:]) if new_docs else "No context yet."
            prompt  = f"{inp}\n\nContext:\n{context}\n\nSummary:"
            try:
                summary_text = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
                tool_results.append(f"[Summarize] → {summary_text[:500]}")
            except Exception as e:
                tool_results.append(f"[Summarize] → Error: {e}")

        return {
            "documents":        new_docs,
            "document_sources": new_sources,
            "tool_results":     tool_results,
            "plan_step_idx":    idx + 1,
            "steps":            state.get("steps", []) + [f"execute_step_{idx}"],
        }

    # ── Targeted retrieve (reflection-driven) ────────────────────────────────

    def _targeted_retrieve(self, state: AgentState) -> Dict:
        """
        NEW: Extra retrieval step triggered by the reflection node when the
        generated answer is incomplete.  Uses the refined query from reflection.
        """
        logger.info("--- TARGETED RETRIEVE (reflection-driven) ---")
        query    = state.get("reflection_query") or state["original_question"]
        texts, sources = self._retrieve_docs(query, settings.TOP_K_RETRIEVAL)

        new_docs    = list(state.get("documents", []))    + texts
        new_sources = list(state.get("document_sources", [])) + sources
        tool_results = list(state.get("tool_results", []))
        tool_results.append(f"[Targeted retrieve: '{query}'] → {len(texts)} chunks.")

        return {
            "documents":        new_docs,
            "document_sources": new_sources,
            "tool_results":     tool_results,
            "steps":            state.get("steps", []) + ["targeted_retrieve"],
        }

    # ── Generate ─────────────────────────────────────────────────────────────

    def _generate(self, state: AgentState) -> Dict:
        """
        Generate final answer from all accumulated context.

        Tool-calling (calculator) is still supported here for any math that
        surfaces during generation.
        """
        logger.info("--- GENERATE (mode=%s, tokens=%d) ---", state["mode"], state["max_tokens"])

        question     = state["original_question"]
        context      = "\n\n".join(state.get("documents", []))
        tool_results = "\n".join(state.get("tool_results", []))
        history      = state.get("chat_history", "")

        if state["mode"] == "detailed":
            system_prompt = (
                f"You are a thorough analyst. Provide a detailed answer using up to "
                f"{state['max_tokens']} tokens. Cover all aspects of the context. "
                f"Do NOT output raw JSON or mention tool names."
            )
        else:
            system_prompt = (
                "You are a concise assistant. Answer directly and briefly. Do not output JSON."
            )

        prompt = f"""{system_prompt}

Research steps completed:
{tool_results}

Retrieved context:
{context}

Chat history:
{history}

Question: {question}
Answer:"""

        writer          = self._writer(state["temperature"], state["max_tokens"])
        writer_w_tools  = writer.bind_tools([calculate])
        messages        = [HumanMessage(content=prompt)]
        response        = writer_w_tools.invoke(messages)

        # Handle tool calls from the LLM (e.g. calculator mid-generation)
        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                if tc["name"] == "calculate":
                    expr = tc["args"].get("expression", "")
                    logger.info("LLM invoked calculator: %s", expr)
                    try:
                        calc_result = calculate.invoke(tc["args"])
                        messages.append(ToolMessage(content=str(calc_result), tool_call_id=tc["id"]))
                    except Exception as e:
                        messages.append(ToolMessage(content=f"Calculation failed: {e}", tool_call_id=tc["id"]))
            response = writer_w_tools.invoke(messages)

        text = response.content
        # Clean up hallucinated JSON blocks that sometimes leak out
        if "```json" in text and "calculate" in text:
            text = text.split("```json")[0].strip()

        return {"generation": text, "steps": state.get("steps", []) + ["generate"]}

    # ── Reflect ──────────────────────────────────────────────────────────────

    def _reflect(self, state: AgentState) -> Dict:
        """
        NEW: LLM critiques its own generated answer.

        Three possible verdicts:
          - good        : answer is complete → route to END
          - needs_more  : answer is incomplete → targeted_retrieve → generate again
          - replan      : question was misunderstood → back to planner
        """
        logger.info("--- REFLECT (round %d) ---", state.get("reflection_count", 0) + 1)
        prompt = f"""You are a quality reviewer for an AI answer.

Question: {state['original_question']}

Generated answer:
{state['generation']}

Retrieved context (first 800 chars):
{chr(10).join(state.get('documents', []))[:800]}

Evaluate the answer:
1. Does it directly answer the question?
2. Are there obvious gaps where more context is needed?
3. Was the question misunderstood entirely?

Return ONLY valid JSON:
{{
  "verdict": "good" | "needs_more" | "replan",
  "reason": "<one sentence>",
  "refined_query": "<only if needs_more: a better search query to fill the gap>"
}}"""

        result  = self._invoke_json(prompt, {"verdict": "good", "reason": "fallback", "refined_query": ""})
        verdict = result.get("verdict", "good")
        reason  = result.get("reason", "")
        refined = result.get("refined_query", "")

        logger.info("Reflection verdict: %s — %s", verdict, reason)
        return {
            "reflection_verdict": verdict,
            "reflection_query":   refined,
            "reflection_count":   state.get("reflection_count", 0) + 1,
            "retry_count":        state.get("retry_count", 0) + (1 if verdict != "good" else 0),
            "steps":              state.get("steps", []) + ["reflect"],
        }

    # -----------------------------------------------------------------------
    # Edge conditions
    # -----------------------------------------------------------------------

    def _route_decision(self, state: AgentState) -> str:
        return state["decision"]

    def _after_step(self, state: AgentState) -> str:
        """Loop back to execute_step until all plan steps are done."""
        if state["plan_step_idx"] < len(state["plan"]):
            return "execute_step"
        return "generate"

    def _after_reflect(self, state: AgentState) -> str:
        """
        Route based on reflection verdict.

        Guard: if we have already reflected MAX_REFLECTIONS times, stop
        regardless of the verdict so we never infinite-loop.
        """
        if state.get("reflection_count", 0) >= self.MAX_REFLECTIONS:
            logger.info("Max reflections reached — forcing END.")
            return "end"

        verdict = state.get("reflection_verdict", "good")
        if verdict == "needs_more":
            return "targeted_retrieve"
        if verdict == "replan":
            return "replan"
        return "end"

    # -----------------------------------------------------------------------
    # Graph construction
    # -----------------------------------------------------------------------

    def _build_graph(self):
        wf = StateGraph(AgentState)

        # Register nodes
        wf.add_node("router",            self._router)
        wf.add_node("chitchat",          self._chitchat)
        wf.add_node("planner",           self._planner)
        wf.add_node("execute_step",      self._execute_step)
        wf.add_node("generate",          self._generate)
        wf.add_node("reflect",           self._reflect)
        wf.add_node("targeted_retrieve", self._targeted_retrieve)

        # Entry point
        wf.set_entry_point("router")

        # router → chitchat | planner
        wf.add_conditional_edges(
            "router",
            self._route_decision,
            {"chitchat": "chitchat", "vectorstore": "planner"},
        )

        # chitchat → done
        wf.add_edge("chitchat", END)

        # planner → first step
        wf.add_edge("planner", "execute_step")

        # execute_step → loop or generate
        wf.add_conditional_edges(
            "execute_step",
            self._after_step,
            {"execute_step": "execute_step", "generate": "generate"},
        )

        # generate → reflect
        wf.add_edge("generate", "reflect")

        # reflect → targeted_retrieve | replan | end
        wf.add_conditional_edges(
            "reflect",
            self._after_reflect,
            {
                "targeted_retrieve": "targeted_retrieve",
                "replan":            "planner",
                "end":               END,
            },
        )

        # targeted_retrieve → generate (one more pass with extra context)
        wf.add_edge("targeted_retrieve", "generate")

        return wf.compile()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def query(
        self,
        query: str,
        mode: str = "concise",
        temperature: float = 0.1,
        max_tokens: int = settings.MAX_TOKENS,
        chat_history: str = "",
    ) -> Dict[str, Any]:
        """
        Run the agentic RAG pipeline.

        Returns:
            {
                "answer":   str,
                "sources":  List[Dict],   ← now populated (was always [] before)
                "metadata": {
                    "steps":        List[str],
                    "plan":         List[Dict],
                    "tool_results": List[str],
                }
            }
        """
        initial: AgentState = {
            "question":           query,
            "original_question":  query,
            "chat_history":       chat_history,
            "mode":               mode,
            "temperature":        temperature,
            "max_tokens":         max_tokens,
            "documents":          [],
            "document_sources":   [],
            "decision":           "vectorstore",
            "generation":         "",
            "plan":               [],
            "plan_step_idx":      0,
            "tool_results":       [],
            "reflection_verdict": "",
            "reflection_query":   "",
            "reflection_count":   0,
            "retry_count":        0,
            "steps":              [],
        }

        result = self._app.invoke(initial)

        return {
            "answer":   result.get("generation", ""),
            "sources":  result.get("document_sources", []),
            "metadata": {
                "steps":        result.get("steps", []),
                "plan":         result.get("plan", []),
                "tool_results": result.get("tool_results", []),
            },
        }