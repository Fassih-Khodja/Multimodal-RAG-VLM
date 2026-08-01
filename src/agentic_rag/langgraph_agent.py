"""
The SAME agent as `learningçagentic_reAct.py`, but rebuilt on LangGraph.

The other file did everything by hand: a while loop, a regex parser, a text
"scratchpad", and a `stop=["Observation:"]` trick to stop the model faking tool
results. That teaches you what an agent *really* is underneath.

This file throws all of that away and lets a framework do it. The goal is to see
EXACTLY which hand-written pieces LangGraph replaces, so we map them 1-to-1:

    HAND-WRITTEN VERSION                    LANGGRAPH VERSION
    ----------------------------------      ---------------------------------------
    while loop in run_agent()          ->   the graph itself (nodes + edges + loop)
    scratchpad string                  ->   State["messages"] (a list of messages)
    "Thought/Action/Action Input" text ->   native tool-calling (structured JSON)
    regex parse_llm_output()           ->   the model returns tool_calls, no parsing
    stop=["Observation:"]              ->   not needed; the API stops after a call
    TOOLS dict + execute_tool()        ->   @tool decorator + ToolNode
    pasting "Observation:" back in     ->   ToolNode appends a ToolMessage for us
    MAX_ITERATIONS safety cap          ->   recursion_limit on the graph
    (nothing — no approval step)       ->   a human_approval node + interrupt()  <-- NEW

The three tools (web_search, save_memory, search_memory), the Chroma vector
memory, and the OpenRouter->Gemini plumbing are all IDENTICAL to the other file.
Only the "agent engine" around them changed.

NEW in this version: a **human-in-the-loop approval gate**. Before the agent is
allowed to run a tool that touches the outside world or writes to memory
(web_search, save_memory), the graph PAUSES and asks you — a real human — to
approve or reject it. This is one of the main reasons frameworks like LangGraph
exist: pausing a running program, waiting for a human, then resuming it exactly
where it left off is genuinely hard to do by hand.

Run it:  python learning_langgraph_agent.py

Install first (into your venv):
    pip install langgraph langchain-openai
"""

import os
import uuid
import textwrap
from datetime import datetime, timezone

import requests
import chromadb
from dotenv import load_dotenv

# --- LangChain / LangGraph imports ------------------------------------------
# langchain_openai.ChatOpenAI  -> an LLM client that speaks the OpenAI API, and
#   therefore also speaks to OpenRouter (which is OpenAI-shaped). This is the
#   framework's replacement for the raw `OpenAI(...)` client in the other file.
# langchain_core.tools.tool    -> the @tool decorator. It reads a function's
#   name, docstring, and type hints and turns them into a JSON schema the model
#   can see. This replaces both our TOOLS dict AND the TOOL_DESCRIPTIONS text.
# langchain_core.messages      -> typed message objects (System/Human/AI/Tool).
#   The list of these IS our new scratchpad.
# langgraph.graph.StateGraph   -> the thing we wire nodes and edges onto.
# langgraph.prebuilt.ToolNode  -> a ready-made node that runs whatever tool the
#   model asked for and appends the result as a ToolMessage. This is
#   execute_tool() + the "paste the Observation back" step, prebuilt.
# langgraph.checkpoint.memory.InMemorySaver -> saves the graph's state after
#   every step. REQUIRED for human-in-the-loop: to pause and later resume, the
#   graph must be able to remember where it was. Here we keep it in RAM.
# langgraph.types.interrupt / Command -> interrupt() pauses the graph and hands
#   a value out to us; Command(resume=...) feeds a value back in to continue.
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Annotated, TypedDict


# =============================================================================
# SECTION 1 — CONFIG & API KEYS   (identical to the hand-written file)
# =============================================================================
# Same .env, same two keys, same OpenRouter->Gemini routing. Nothing here is
# LangGraph-specific — the framework doesn't care where the model lives.
# -----------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Put it in a .env file (see SECTION 1).")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Put it in a .env file (see SECTION 1).")

LLM_MODEL = "google/gemini-2.5-flash"
CHROMA_PATH = "./agent_memory_db"
MEMORY_COLLECTION = "long_term_memory"
SEARCH_RESULT_CHAR_LIMIT = 500

# LangGraph's equivalent of MAX_ITERATIONS. The graph loops agent->tools->agent;
# recursion_limit caps how many node-steps a single run may take before it
# raises, so a broken agent can't loop (and bill you) forever. Each full
# agent->approval->tools cycle is a few steps, so we give it some headroom.
RECURSION_LIMIT = 25

# Which tools require a human to approve them before they run. Reading memory is
# harmless, so it's auto-approved. Writing memory or hitting the live internet
# are the actions we want a human to sign off on.
TOOLS_NEEDING_APPROVAL = {"web_search", "save_memory"}


# =============================================================================
# SECTION 2 — LOGGING HELPERS   (copied verbatim so the trace looks the same)
# =============================================================================

class C:
    """ANSI color codes, so the different parts of the trace are easy to tell apart."""
    GREY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    OFF = "\033[0m"


def section(title):
    print(f"\n{C.BOLD}{C.BLUE}{'=' * 70}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}  {title}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}{'=' * 70}{C.OFF}")


def log(tag, message, color=C.GREY):
    body = textwrap.indent(str(message), " " * 14).strip()
    print(f"{color}{C.BOLD}[{tag:^12}]{C.OFF} {color}{body}{C.OFF}")


def log_block(tag, message, color=C.GREY):
    print(f"{color}{C.BOLD}[{tag:^12}]{C.OFF} {color}┌{'─' * 58}{C.OFF}")
    for line in str(message).splitlines() or [""]:
        print(f"{color}               │ {line}{C.OFF}")
    print(f"{color}               └{'─' * 58}{C.OFF}")


# =============================================================================
# SECTION 3 — THE LLM   (now a LangChain ChatOpenAI object, not a raw client)
# =============================================================================
# In the other file we called client.chat.completions.create(...) ourselves and
# read response.choices[0].message.content. Here the framework wraps all that.
#
# The single most important line below is `.bind_tools(...)` further down. In the
# hand-written file we could NOT use native tool-calling, so we invented the
# whole "Thought/Action/Action Input" text protocol and parsed it with a regex.
# Here, bind_tools tells the model — via the API's real function-calling feature
# — exactly which tools exist and their argument schemas. The model then replies
# with a *structured* tool call (clean JSON), so there is nothing to parse and no
# stop=["Observation:"] hack needed. That entire class of fragility disappears.
# -----------------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url="https://openrouter.ai/api/v1",   # same OpenRouter pipe as before
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,                            # deterministic, reproducible traces
    max_tokens=1024,
)


# =============================================================================
# SECTION 4 — LONG-TERM MEMORY   (identical Chroma setup)
# =============================================================================
# Byte-for-byte the same idea as the other file: embed text into vectors, store
# them, and search by nearest-neighbour. One difference: LangChain's ChatOpenAI
# is a *chat* client and doesn't do embeddings, so we still call OpenRouter's
# embeddings endpoint through the plain `openai` SDK, exactly like before.
# -----------------------------------------------------------------------------

from openai import OpenAI  # only used for the embeddings call below

embed_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
EMBED_MODEL = "google/gemini-embedding-001"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
memory_collection = chroma_client.get_or_create_collection(
    name=MEMORY_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


def embed(text, task_type):
    """Turn text into a vector via Gemini's embedding model (through OpenRouter)."""
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    vector = response.data[0].embedding
    log("EMBED", f"{task_type}: '{text[:45]}...' → vector of {len(vector)} floats", C.MAGENTA)
    return vector


# =============================================================================
# SECTION 5 — THE TOOLS   (same logic, now wearing the @tool decorator)
# =============================================================================
# Compare this to the other file's tools. The BODIES are the same. What changed:
#
#   1. Each function is decorated with @tool. That decorator inspects the
#      function name, the docstring, and the type hints and auto-generates the
#      JSON schema the model needs. So the docstring is NOT just a comment here —
#      the model literally reads it to decide when to call the tool. This is why
#      we no longer need the separate TOOL_DESCRIPTIONS block from the old file:
#      the description now lives on the function itself.
#
#   2. The argument is now a typed parameter (query: str) instead of a bare
#      string. The type hint becomes part of the schema the model sees.
#
# Everything else — Tavily REST call, Chroma add/query — is unchanged.
# -----------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the live internet. Use for current events, facts you don't know, or
    anything that may have changed recently. Input: a search query string."""
    log("TOOL", f"web_search(query='{query}')", C.YELLOW)

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3,
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not results:
        return "No results found for that query."

    lines = []
    for i, r in enumerate(results, start=1):
        content = r.get("content", "")[:SEARCH_RESULT_CHAR_LIMIT]
        lines.append(f"[{i}] {r.get('title', 'untitled')}\n    {content}\n    (source: {r.get('url', '')})")

    output = "\n".join(lines)
    log("TOOL", f"web_search returned {len(results)} results ({len(output)} chars)", C.YELLOW)
    return output


@tool
def save_memory(text: str) -> str:
    """Save ONE durable fact to long-term memory so it survives across
    conversations. Input: the fact to remember, written as a complete sentence."""
    log("TOOL", f"save_memory(text='{text}')", C.YELLOW)

    vector = embed(text, task_type="RETRIEVAL_DOCUMENT")
    memory_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[text],
        metadatas=[{"timestamp": datetime.now(timezone.utc).isoformat(), "source": "agent"}],
    )
    log("TOOL", f"saved. collection now holds {memory_collection.count()} memories", C.YELLOW)
    return f"Saved to long-term memory: '{text}'"


@tool
def search_memory(query: str) -> str:
    """Look up facts previously saved to long-term memory. Input: a description of
    what you're looking for."""
    log("TOOL", f"search_memory(query='{query}')", C.YELLOW)

    if memory_collection.count() == 0:
        return "Long-term memory is empty. Nothing has been saved yet."

    vector = embed(query, task_type="RETRIEVAL_QUERY")
    results = memory_collection.query(
        query_embeddings=[vector],
        n_results=min(3, memory_collection.count()),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    lines = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist
        lines.append(f"- {doc}  (saved: {meta.get('timestamp', '?')[:10]}, similarity: {similarity:.2f})")
        log("MEMORY", f"hit: '{doc[:40]}...' similarity={similarity:.2f}", C.MAGENTA)

    return "Found these memories:\n" + "\n".join(lines)


# The tool registry. In the old file TOOLS was a dict we looked up by hand. Here
# it's just a plain list we hand to two places: the LLM (so it knows the schemas)
# and the ToolNode (so it can actually run them). Same list, two consumers.
TOOLS = [web_search, save_memory, search_memory]

# THIS is the line that replaces the entire regex/stop-sequence protocol. It
# attaches the tools' JSON schemas to the model. From now on `llm_with_tools`
# can answer with either normal text OR a structured request to call a tool.
llm_with_tools = llm.bind_tools(TOOLS)

# A name->tool map, used only by our human_approval node to print a nice summary
# of what's about to run. ToolNode does its own lookup internally.
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# =============================================================================
# SECTION 6 — THE SYSTEM PROMPT   (much shorter now — the format is handled)
# =============================================================================
# The old system prompt spent most of its length TEACHING THE OUTPUT GRAMMAR:
# "reply using this exact format: Thought/Action/Action Input... NEVER write an
# Observation yourself...". All of that is gone. With native tool-calling the API
# guarantees the structure, so the prompt only has to convey *policy* — when to
# use memory vs. the web — not *syntax*. This shrinkage is a direct benefit of
# letting the framework own the protocol.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful research assistant that reasons step by step
and uses tools when they help.

RULES FOR MEMORY (think about these; do not use memory reflexively):
- Call search_memory FIRST when the question refers to the user personally, to
  things you were told before, or to anything you might already have been told
  ("what do you know about me", "what's my favourite...", "where do I work").
- Call save_memory when the user tells you a durable fact worth remembering
  later: preferences, personal details, decisions, ongoing projects.
- Do NOT save trivia, small talk, or facts you looked up on the web.
- Do NOT call a tool at all if you can already answer. Simple questions
  ("what is 2+2") should be answered directly.

When you have enough information, reply to the user in plain language with no
tool call — that plain reply is treated as your final answer."""


# =============================================================================
# SECTION 7 — THE GRAPH STATE
# =============================================================================
# In the hand-written agent the ENTIRE state was one growing string called
# `scratchpad`. Here the state is a dict with one key: a list of messages.
#
# The `Annotated[list, add_messages]` part is the important bit. `add_messages`
# is a "reducer": when a node returns some messages, LangGraph does NOT overwrite
# the list — it APPENDS them (and is smart about updating messages by id). So
# every node just returns "the new message(s) I produced" and the framework
# accumulates the running transcript for us. That accumulation is precisely what
# our `scratchpad += ...` lines did by hand.
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =============================================================================
# SECTION 8 — THE NODES
# =============================================================================
# A LangGraph "node" is just a function: it receives the current state and
# returns a dict of updates to the state. We have three nodes:
#
#   agent_node          — calls the LLM; may produce a tool call
#   human_approval_node — pauses for a human before a risky tool runs   (NEW!)
#   tool_node           — runs the approved tool (this is the prebuilt ToolNode)
#
# The edges between them (Section 9) recreate the old while-loop.
# -----------------------------------------------------------------------------

def agent_node(state: AgentState) -> dict:
    """
    The "brain" step. Equivalent to STEP 1 (call_llm) + STEP 2 (parse) of the old
    loop combined — except there's no parsing, because the model returns a proper
    AIMessage. That message either:
        - has .tool_calls  -> the model wants to run a tool, OR
        - has plain .content and no tool_calls -> that's the final answer.

    We send the whole message history every time (llm_with_tools.invoke(messages)).
    That is the exact same idea as re-sending the whole scratchpad each iteration:
    the model is stateless, the message list is the memory.
    """
    messages = state["messages"]
    log("AGENT", f"→ sending {len(messages)} messages to {LLM_MODEL}", C.CYAN)

    ai_message = llm_with_tools.invoke(messages)

    if ai_message.tool_calls:
        # The model chose to act. Log each requested call (name + arguments).
        for tc in ai_message.tool_calls:
            log("AGENT", f"model wants tool: {tc['name']}({tc['args']})", C.CYAN)
    else:
        # No tool call = the model is answering directly. This is our "final".
        log("AGENT", "model produced a direct answer (no tool call)", C.GREEN)
        log_block("RAW LLM", ai_message.content, C.GREY)

    # Returning it under "messages" -> add_messages APPENDS it to the transcript.
    return {"messages": [ai_message]}


def human_approval_node(state: AgentState) -> Command:
    """
    THE NEW PIECE: a human-in-the-loop gate. *** This is the star of this file. ***

    This node runs AFTER the agent asked for a tool but BEFORE the tool executes.
    It looks at the pending tool call and, if that tool is in
    TOOLS_NEEDING_APPROVAL, it PAUSES the entire graph and waits for a human.

    How the pause works — `interrupt(payload)`:
        Calling interrupt() throws a special signal that LangGraph catches. The
        graph stops dead, saves its state to the checkpointer, and the value you
        passed to interrupt() is handed back to the code that ran the graph (see
        the driver in Section 10). Your program is now free to do anything —
        print a prompt, read input(), send a Slack message, wait a week — and
        then RESUME the graph by re-invoking it with Command(resume=<answer>).
        When it resumes, this node runs AGAIN from the top, but this time the
        interrupt() call *returns* the value you resumed with instead of pausing.

        Doing this by hand in the old while-loop would have been genuinely nasty:
        you'd have to serialize the scratchpad, exit, and rebuild the loop state
        on restart. The checkpointer + interrupt() gives it to us for free.

    We return a `Command` telling the graph which node to go to next:
        - approved -> goto "tools"  (let it run)
        - rejected -> goto "agent"  (and we inject a ToolMessage saying the human
                      refused, so the model can react — apologise, try something
                      else, or just answer without that tool).
    """
    last_message = state["messages"][-1]
    # There could in theory be several tool calls; this teaching agent handles
    # them one at a time, so we look at the first.
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Safe, read-only tools skip the gate entirely and go straight to execution.
    if tool_name not in TOOLS_NEEDING_APPROVAL:
        log("APPROVAL", f"'{tool_name}' is auto-approved (read-only) → running", C.GREEN)
        return Command(goto="tools")

    log("APPROVAL", f"'{tool_name}' needs human sign-off — PAUSING the graph", C.RED)

    # This dict is the payload handed out to the human. In a real app it would be
    # rendered in a UI; here the driver prints it and calls input().
    decision = interrupt({
        "tool": tool_name,
        "args": tool_args,
        "question": f"Allow the agent to run {tool_name} with these arguments?",
    })

    # --- Everything below here runs only AFTER a human resumed the graph. ------
    log("APPROVAL", f"human responded: {decision!r}", C.BLUE)

    approved = str(decision).strip().lower() in {"y", "yes", "approve", "approved", "ok", "true"}

    if approved:
        log("APPROVAL", "APPROVED → proceeding to run the tool", C.GREEN)
        return Command(goto="tools")

    # Rejected. We must still add a ToolMessage answering this tool_call (the API
    # requires every tool_call to be answered), but the "answer" is a note that a
    # human blocked it. Then we send control back to the agent to rethink.
    log("APPROVAL", "REJECTED → telling the agent a human blocked the tool", C.RED)
    rejection = ToolMessage(
        content=(
            f"A human reviewer REJECTED the call to '{tool_name}'. "
            f"Do not retry it. Either answer the user without this tool, or explain "
            f"that you cannot proceed."
        ),
        tool_call_id=tool_call["id"],
    )
    return Command(goto="agent", update={"messages": [rejection]})


# The prebuilt ToolNode. Hand it the same TOOLS list and it does, automatically,
# what our old execute_tool() + "paste the Observation back" did: it reads the
# tool_calls off the last AIMessage, runs each tool, catches errors, and appends
# a ToolMessage (the "Observation") for each result. No dict lookup, no
# try/except of our own to write.
tool_node = ToolNode(TOOLS)


# =============================================================================
# SECTION 9 — WIRING THE GRAPH (edges = the control flow of the old while-loop)
# =============================================================================
# The old run_agent() encoded its control flow in Python `if`s inside a for-loop.
# LangGraph makes that flow explicit as a graph of nodes and edges:
#
#         START
#           │
#           ▼
#        ┌──────┐   has tool_calls?   ┌───────────┐   approved   ┌───────┐
#        │agent │ ──────yes────────▶  │  approval │ ───────────▶ │ tools │
#        └──────┘                     └───────────┘              └───────┘
#           ▲   │ no                       │ rejected                │
#           │   ▼                          │                         │
#           │  END                         └───────┐   ┌─────────────┘
#           └──────────────────────────────────────┴───┘  (both loop back to agent)
#
# `should_continue` is the one conditional edge we still write by hand: after the
# agent speaks, does it want a tool (go to approval) or is it done (go to END)?
# This is the direct replacement for the `if kind == "final"` check.
# -----------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    """Router: look at the agent's last message. Tool call -> approval; else END."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        log("ROUTER", "agent requested a tool → going to approval gate", C.BLUE)
        return "human_approval"
    log("ROUTER", "agent gave a final answer → ending run", C.GREEN)
    return END


def build_graph():
    """Assemble the nodes and edges into a runnable, compiled graph."""
    builder = StateGraph(AgentState)

    # 1. Register the three nodes under names we'll refer to in edges.
    builder.add_node("agent", agent_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("tools", tool_node)

    # 2. The run always starts at the agent.
    builder.add_edge(START, "agent")

    # 3. After the agent, branch on should_continue's return value.
    builder.add_conditional_edges("agent", should_continue, ["human_approval", END])

    # 4. human_approval returns a Command(goto=...) so it picks its own next node
    #    dynamically — no static edge needed out of it. But we DO list its
    #    possible destinations so LangGraph can validate the graph.
    #    (tools and agent are already nodes; nothing to add here.)

    # 5. After a tool runs, always loop back to the agent so it can read the
    #    Observation and decide the next step. This closes the loop.
    builder.add_edge("tools", "agent")

    # 6. Compile. The checkpointer is REQUIRED for interrupt()/resume to work:
    #    it's what lets the graph freeze its state on pause and thaw it on resume.
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph


# =============================================================================
# SECTION 10 — THE DRIVER (runs the graph + handles the human approval pause)
# =============================================================================
# This replaces run_agent(). Because the graph can PAUSE for approval, running it
# is a small loop:
#
#   1. Start (or resume) the graph.
#   2. If it finished -> print the final answer, done.
#   3. If it paused (an interrupt) -> ask the human, then resume with their
#      answer and go back to step 2.
#
# The `thread_id` is how the checkpointer knows which conversation's saved state
# to reload when we resume. Same thread_id = same paused run continues.
# -----------------------------------------------------------------------------

def run_agent(graph, question):
    section(f"NEW RUN — {question}")

    # A fresh thread per question = a fresh, independent conversation state.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}

    # The very first input: the system prompt + the user's question. On resumes we
    # pass a Command instead (see below), so `graph_input` changes each pass.
    graph_input = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]}

    step = 0
    while True:
        step += 1
        print(f"\n{C.BOLD}{C.GREEN}───── GRAPH TURN {step} ─────{C.OFF}")

        # invoke() runs the graph until it either finishes OR hits an interrupt().
        result = graph.invoke(graph_input, config=config)

        # LangGraph reports a pause by putting an "__interrupt__" key in the
        # result. If it's absent, the graph ran to completion.
        if "__interrupt__" not in result:
            final = result["messages"][-1].content
            section("FINAL ANSWER")
            print(f"{C.BOLD}{C.GREEN}{final}{C.OFF}\n")
            return final

        # --- We were paused by the human_approval node. --------------------
        # result["__interrupt__"] is a list of Interrupt objects; .value is the
        # payload dict we passed to interrupt() back in the node.
        interrupt_payload = result["__interrupt__"][0].value

        section("⏸  HUMAN APPROVAL REQUIRED")
        log("REVIEW", interrupt_payload["question"], C.YELLOW)
        log("REVIEW", f"tool: {interrupt_payload['tool']}", C.YELLOW)
        log_block("ARGS", interrupt_payload["args"], C.YELLOW)

        answer = input(f"{C.BOLD}{C.MAGENTA}   Approve? [y/N]: {C.OFF}").strip()

        # Command(resume=answer) feeds `answer` back in as the return value of the
        # interrupt() call inside the node, and the graph continues from there.
        graph_input = Command(resume=answer)
        # loop again -> graph.invoke resumes -> may finish or pause once more.


# =============================================================================
# DEMO — the same three runs as the hand-written file
# =============================================================================
# RUN 1 wants save_memory   -> will PAUSE for your approval (save is risky).
# RUN 2 wants search_memory -> auto-approved, no pause (read-only).
# RUN 3 wants web_search    -> will PAUSE for your approval (hits the internet).
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    section("SETUP")
    log("CONFIG", f"llm={LLM_MODEL}  embeddings={EMBED_MODEL} (via OpenRouter)", C.CYAN)
    log("CONFIG", f"chroma path={CHROMA_PATH}  memories stored={memory_collection.count()}", C.CYAN)
    log("CONFIG", f"tools registered: {[t.name for t in TOOLS]}", C.CYAN)
    log("CONFIG", f"tools needing human approval: {sorted(TOOLS_NEEDING_APPROVAL)}", C.CYAN)

    agent_graph = build_graph()
    log("CONFIG", "graph compiled: agent → human_approval → tools → agent", C.CYAN)

    # RUN 1 — should call save_memory (a durable personal fact) -> PAUSES.
    run_agent(agent_graph, "Remember that I'm building a vision-language model side project, and that I prefer explanations with lots of comments.")

    # RUN 2 — should call search_memory (asks about the user) -> no pause.
    run_agent(agent_graph, "What do you know about my side project and how I like explanations?")

    # RUN 3 — should call web_search (a fact the model can't know) -> PAUSES.
    run_agent(agent_graph, "what is the most lovable language by people in the world of course after english?")

    section("DONE")
    log("CONFIG", f"long-term memory now holds {memory_collection.count()} facts", C.CYAN)
    log("CONFIG", f"they persist in {CHROMA_PATH}/ — run again and they'll still be there", C.CYAN)
