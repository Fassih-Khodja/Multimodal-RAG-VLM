"""
============================================================================
  learning_vlm_rag_agent.py
  A LangGraph AGENT wrapped around YOUR existing VLM-RAG pipeline.
============================================================================

WHAT THIS IS
------------
Your project already has a *linear* RAG pipeline:

    prompt ──▶ search_vlm() ──▶ ask_openrouter()  (attach chunks + images) ──▶ answer

That pipeline ALWAYS does the same thing in the same order. It cannot decide
"this question isn't in the PDF, I should search the web instead", and it can't
choose to write a study note for you. It just retrieves and answers.

This file turns that pipeline into an AGENT. An agent is a loop where an LLM
looks at the situation and *decides what to do next* — call a tool, call another
tool, or stop and answer. Nothing about your retrieval or your VLM changes; we
just put a decision-making brain on top of them.

THE THREE TOOLS THE AGENT CAN CHOOSE FROM
-----------------------------------------
  1. retrieve_from_document(query)  -> reuses YOUR search_vlm() from
       search_qdrant.py. Pulls the top text chunks AND the relevant page images
       out of Qdrant. The images are stashed and shown to the VLM on the next
       thinking step, so the model literally *sees* the figures — same as
       ask_vlm.py did, but now on demand.
  2. web_search(query)             -> Tavily live web search (same call your
       other learning files use). For things that aren't in the attached PDF.
  3. write_study_summary(...)      -> writes a Markdown study note to
       study_notes/. *** This one PAUSES and asks YOU to approve *** before it
       touches your disk (human-in-the-loop).

HOW THE AGENT DECIDES (this is the part you asked for)
------------------------------------------------------
The model reads the prompt and picks a path on its own:

    ┌─ Is the answer likely IN the attached document? ──▶ retrieve_from_document
    │                                                        │
    │                                                        ▼
    │                                          read chunks + SEE the images
    │                                                        │
    │                          enough to answer? ──▶ answer directly
    │                          still missing facts? ──▶ web_search, then answer
    │                          asked for a study note? ──▶ write_study_summary (approval)
    │
    └─ Is it clearly a live/world question not in the PDF? ──▶ web_search directly

It can also do BOTH (retrieve then web_search) in one run, because after every
tool result the loop goes back to the brain and it decides again.

WHAT YOU ALREADY KNOW MAPS LIKE THIS (vs. learning_langgraph_agent.py)
----------------------------------------------------------------------
Same graph skeleton (agent → approval → tools → agent) and same human-approval
idea. Two things are genuinely NEW here and worth studying:
  • A tool (retrieve_from_document) that returns a `Command` to update graph
    state — that's how the retrieved IMAGES get carried to the next step.
  • The agent node injecting a multimodal (image) message into the transcript,
    so a text-first agent loop can still do VISION. This is the bit that
    reconnects your VLM to the agent.

PREREQUISITES
-------------
  • Qdrant running with your collections already ingested (same as ask_vlm.py):
        docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
    If Qdrant is down, retrieve_from_document fails gracefully and the agent is
    told to fall back to web_search — a nice thing to watch happen.
  • .env with GEMINI_API_KEY (your OpenRouter key) and TAVILY_API_KEY.

RUN
---
    python learning_vlm_rag_agent.py
"""

import os
import re
import base64
import textwrap
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import requests
from PIL import Image
from dotenv import load_dotenv

# --- LangChain / LangGraph -------------------------------------------------
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# --- YOUR existing pipeline ------------------------------------------------
# We import search_vlm DIRECTLY from your file. This is the whole point of the
# exercise: the agent reuses your real retrieval, it does not reimplement it.
# (We deliberately do NOT import ask_vlm.py, because that module raises at import
#  time if OPENROUTER_API_KEY isn't set. We only need its tiny image helper, so
#  we re-create that one function below with attribution.)
from search_qdrant import search_vlm


# =============================================================================
# SECTION 1 — CONFIG & API KEYS
# =============================================================================
# Note the same trick your learning_langgraph_agent.py uses: your .env stores an
# OpenRouter key under the name GEMINI_API_KEY, and OpenRouter speaks the OpenAI
# API, so we point an OpenAI-shaped client at OpenRouter's base_url.
# -----------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("GEMINI_API_KEY (your OpenRouter key) is not set in .env")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in .env")

# Must be a VISION-capable model, because we feed it page images. gemini-2.5-flash
# is multimodal and cheap. (Your ask_vlm.py used openai/gpt-4o — also fine here.)
LLM_MODEL = "google/gemini-2.5-flash"

STUDY_NOTES_DIR = "study_notes"          # where write_study_summary saves files
SEARCH_RESULT_CHAR_LIMIT = 500           # trim each web result so context stays small
IMAGE_MAX_SIZE = (800, 800)              # same downscale as ask_vlm.py
RECURSION_LIMIT = 30                     # safety cap on graph node-steps per run

# Only the disk-writing tool needs your sign-off. Retrieval and web search are
# read-only, so they run without interrupting you. (You could add "web_search"
# here too if you want to approve every outbound network call.)
TOOLS_NEEDING_APPROVAL = {"write_study_summary"}


# =============================================================================
# SECTION 2 — LOGGING HELPERS (so the whole decision trace is readable)
# =============================================================================

class C:
    """ANSI colours to tell the parts of the trace apart."""
    GREY = "\033[90m"; RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"; BOLD = "\033[1m"; OFF = "\033[0m"


def section(title):
    print(f"\n{C.BOLD}{C.BLUE}{'=' * 72}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}  {title}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}{'=' * 72}{C.OFF}")


def log(tag, message, color=C.GREY):
    body = textwrap.indent(str(message), " " * 14).strip()
    print(f"{color}{C.BOLD}[{tag:^12}]{C.OFF} {color}{body}{C.OFF}")


def log_block(tag, message, color=C.GREY):
    print(f"{color}{C.BOLD}[{tag:^12}]{C.OFF} {color}┌{'─' * 58}{C.OFF}")
    for line in str(message).splitlines() or [""]:
        print(f"{color}               │ {line}{C.OFF}")
    print(f"{color}               └{'─' * 58}{C.OFF}")


# =============================================================================
# SECTION 3 — THE LLM (a LangChain ChatOpenAI pointed at OpenRouter)
# =============================================================================
# .bind_tools() below is what makes this an agent: it hands the model the JSON
# schemas of our tools (auto-generated from each @tool's name/docstring/type
# hints). The model can then answer with plain text OR a structured request to
# call one of the tools. No regex, no "Thought/Action" text protocol.
# -----------------------------------------------------------------------------

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,      # deterministic so your traces are reproducible while learning
    max_tokens=1200,
)


# =============================================================================
# SECTION 4 — SMALL IMAGE HELPER (copied from your ask_vlm.py, verbatim logic)
# =============================================================================
# Kept local so this file doesn't import ask_vlm (which would crash on missing
# OPENROUTER_API_KEY). Same resize + base64 as your original.
# -----------------------------------------------------------------------------

def get_base64_image(image_path: str, max_size=IMAGE_MAX_SIZE) -> str:
    """Read an image, downscale it, return a base64 JPEG string."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# =============================================================================
# SECTION 5 — THE GRAPH STATE
# =============================================================================
# Same idea as learning_langgraph_agent.py, PLUS one new field.
#
#   messages        : the running transcript (add_messages APPENDS to it).
#   pending_images  : image file paths that retrieve_from_document just found but
#                     the VLM hasn't SEEN yet. The agent node consumes this on its
#                     next turn (attaches the images to the model call) and clears
#                     it. This is the bridge that carries images from a text tool
#                     result into a vision model call.
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_images: list      # last-write-wins (tool sets it, agent clears it)


# =============================================================================
# SECTION 6 — THE TOOLS
# =============================================================================

# ---- internal helper used by the retrieval tool ----------------------------
def _search_and_collect(query: str):
    """
    Run YOUR search_vlm() and turn its result into (context_text, image_paths).
    This is exactly the context-assembly logic from ask_vlm.ask_openrouter(),
    lifted out so the tool can reuse it.

    Returns:
        context_text : str            — the retrieved chunks, formatted
        image_paths  : list[str]      — de-duplicated image files to show the VLM
    """
    results = search_vlm(prompt=query)   # <-- your function, unchanged

    # (a) format the retrieved text chunks
    blocks = []
    for i, chunk in enumerate(results["chunks"]):
        text = chunk.get("text", "")
        if text:
            blocks.append(f"[Chunk {i + 1} | score={chunk.get('score'):.3f}]\n{text}")
    context_text = "\n\n".join(blocks) if blocks else "No relevant text chunks were found."

    # (b) collect every relevant image path (same two sources as ask_vlm.py)
    image_paths = set()
    for img in results["images"]:                       # directly retrieved images
        p = img.get("image_path")
        if p and Path(p).exists():
            image_paths.add(p)
    for chunk in results["chunks"]:                     # images sitting next to chunks
        for img_id in chunk.get("nearby_images", []):
            p = f"extracted/images/{img_id}.png"
            if Path(p).exists():
                image_paths.add(p)

    return context_text, sorted(image_paths)


@tool
def retrieve_from_document(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve relevant passages AND figures/images from the attached PDF/manual
    that was ingested into the vector database. Use this FIRST whenever the
    question is about the content of the attached document, a manual, a diagram,
    a figure, a table, or 'what does the document/PDF say'. Input: a focused
    search query describing what to find."""
    # `tool_call_id` is injected by LangGraph (the model never fills it in); we
    # need it to build the ToolMessage that answers this specific tool call.
    log("TOOL", f"retrieve_from_document(query='{query}')", C.YELLOW)

    try:
        context_text, image_paths = _search_and_collect(query)
    except Exception as e:
        # Qdrant down / collection missing / etc. Tell the agent so it can fall
        # back to web_search instead of crashing the whole run.
        msg = (f"Retrieval failed ({type(e).__name__}: {e}). The document store "
               f"may be unavailable. Consider answering from web_search instead.")
        log("TOOL", msg, C.RED)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    log("TOOL", f"retrieved {len(image_paths)} image(s); "
                f"{len(context_text)} chars of text", C.YELLOW)
    for p in image_paths:
        log("IMAGE", f"queued for the VLM to see → {p}", C.MAGENTA)

    # A tool normally just returns a string. Here we return a Command so we can
    # update TWO things at once: the ToolMessage (the text the model reads) AND
    # pending_images (the files the agent node will attach on its next turn).
    tool_reply = (
        f"{context_text}\n\n"
        f"[{len(image_paths)} image(s) from the document have been attached for you "
        f"to look at in your next step.]"
        if image_paths else
        f"{context_text}\n\n[No images were associated with these passages.]"
    )
    return Command(update={
        "messages": [ToolMessage(content=tool_reply, tool_call_id=tool_call_id)],
        "pending_images": image_paths,
    })


@tool
def web_search(query: str) -> str:
    """Search the live internet with Tavily. Use for current events, general
    world knowledge, or anything NOT contained in the attached document. Input: a
    search query string."""
    log("TOOL", f"web_search(query='{query}')", C.YELLOW)

    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query,
              "max_results": 3, "search_depth": "basic"},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return "No web results found for that query."

    lines = []
    for i, r in enumerate(results, start=1):
        content = r.get("content", "")[:SEARCH_RESULT_CHAR_LIMIT]
        lines.append(f"[{i}] {r.get('title', 'untitled')}\n    {content}\n    (source: {r.get('url', '')})")
    out = "\n".join(lines)
    log("TOOL", f"web_search returned {len(results)} results", C.YELLOW)
    return out


@tool
def write_study_summary(topic: str, summary_markdown: str) -> str:
    """Save a study note to disk as a Markdown file so the user can revise later.
    Call this ONLY when the user asks you to write / save / make a summary or study
    note. Inputs: `topic` = a short title for the note; `summary_markdown` = the
    full study summary written in Markdown (headings, bullet points, key terms).
    This action writes a file and requires human approval before it runs."""
    # By the time this BODY runs, the human already approved it (see the approval
    # node). So here we just do the write.
    log("TOOL", f"write_study_summary(topic='{topic}', {len(summary_markdown)} chars)", C.YELLOW)

    Path(STUDY_NOTES_DIR).mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "note"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = Path(STUDY_NOTES_DIR) / f"{stamp}_{slug}.md"

    header = f"# {topic}\n\n_Study note generated on {datetime.now():%Y-%m-%d %H:%M}_\n\n---\n\n"
    file_path.write_text(header + summary_markdown, encoding="utf-8")

    log("TOOL", f"wrote study note → {file_path}", C.GREEN)
    return f"Study note saved to '{file_path}'. Tell the user it's ready."


TOOLS = [retrieve_from_document, web_search, write_study_summary]
llm_with_tools = llm.bind_tools(TOOLS)      # <-- attaches the schemas to the model
tool_node = ToolNode(TOOLS)                 # <-- runs whichever tool the model picked


# =============================================================================
# SECTION 7 — THE SYSTEM PROMPT (this is where you TEACH the agent its policy)
# =============================================================================
# Native tool-calling handles the *format*, so the prompt only carries *policy*:
# when to retrieve vs. search vs. write. Tweaking this text is the main knob you
# have for changing the agent's behaviour — experiment with it.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a study assistant for a technical document (a PDF/manual)
that has been indexed into a vector database. You have three tools.

HOW TO DECIDE WHAT TO DO:
1. If the question is about the CONTENT of the attached document (a concept,
   figure, diagram, table, procedure, "what does it say about X"), call
   retrieve_from_document FIRST. You will get text passages, and any relevant
   images will be shown to you so you can read the figures yourself.
2. After retrieving, judge whether it is enough:
   - Enough  -> answer the user directly, citing what you saw.
   - Missing outside/current facts -> ALSO call web_search, then answer.
   - Retrieval failed or clearly irrelevant -> use web_search instead.
3. If the question is obviously about the live world / current events and not the
   document, call web_search directly.
4. Call write_study_summary ONLY when the user explicitly asks you to write, save,
   or make a summary / study note. First gather the material (retrieve and/or
   search), THEN call write_study_summary with a clear Markdown summary.
5. If you can answer a trivial question with no tool, just answer.

When you have enough information, reply in plain language with NO tool call — that
plain reply is your final answer. Be concrete and cite whether each fact came from
the document or the web."""


# =============================================================================
# SECTION 8 — THE NODES
# =============================================================================

def agent_node(state: AgentState) -> dict:
    """
    The brain. Calls the LLM on the whole transcript and returns its reply.

    THE VISION BRIDGE: if a previous retrieval left images in `pending_images`,
    we build a multimodal HumanMessage (text + base64 images) and slip it in
    BEFORE calling the model, so the VLM actually sees the figures. Then we clear
    pending_images. This is how a text-first agent loop does vision on demand —
    the same image-attaching that ask_vlm.py did, but only when retrieval
    produced images.
    """
    messages = list(state["messages"])
    pending = state.get("pending_images") or []

    injected = []
    if pending:
        log("VISION", f"attaching {len(pending)} retrieved image(s) to the model call", C.MAGENTA)
        content = [{"type": "text",
                    "text": "Here are the image(s) retrieved from the document. "
                            "Look at them carefully; they are part of the context."}]
        for p in pending:
            try:
                b64 = get_base64_image(p)
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            except Exception as e:
                log("VISION", f"could not load {p}: {e}", C.RED)
        img_msg = HumanMessage(content=content)
        injected = [img_msg]
        messages = messages + injected

    log("AGENT", f"→ sending {len(messages)} messages to {LLM_MODEL}", C.CYAN)
    ai_message = llm_with_tools.invoke(messages)

    if ai_message.tool_calls:
        for tc in ai_message.tool_calls:
            log("AGENT", f"model wants tool: {tc['name']}({tc['args']})", C.CYAN)
    else:
        log("AGENT", "model produced a direct answer (no tool call)", C.GREEN)
        log_block("RAW LLM", ai_message.content, C.GREY)

    # Persist the injected image message too (so it stays in history), then the
    # AI reply. Clear pending_images so we don't re-attach the same images.
    return {"messages": injected + [ai_message], "pending_images": []}


def human_approval_node(state: AgentState) -> Command:
    """
    Human-in-the-loop gate. Runs after the agent asked for a tool, before the
    tool executes. Read-only tools pass straight through; a tool in
    TOOLS_NEEDING_APPROVAL PAUSES the graph via interrupt() and waits for you.

    interrupt(payload) freezes the graph (state saved by the checkpointer) and
    hands `payload` out to the driver (Section 9), which shows it to you and
    resumes with Command(resume=<your answer>). On resume this node re-runs and
    interrupt() RETURNS your answer instead of pausing again.
    """
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]
    name, args = tool_call["name"], tool_call["args"]

    if name not in TOOLS_NEEDING_APPROVAL:
        log("APPROVAL", f"'{name}' is read-only → auto-approved", C.GREEN)
        return Command(goto="tools")

    log("APPROVAL", f"'{name}' writes to disk — PAUSING for your approval", C.RED)
    decision = interrupt({
        "tool": name,
        "args": args,
        "question": f"Allow the agent to run {name}?",
    })

    log("APPROVAL", f"human responded: {decision!r}", C.BLUE)
    approved = str(decision).strip().lower() in {"y", "yes", "approve", "approved", "ok", "true"}
    if approved:
        log("APPROVAL", "APPROVED → running the tool", C.GREEN)
        return Command(goto="tools")

    log("APPROVAL", "REJECTED → telling the agent a human blocked it", C.RED)
    rejection = ToolMessage(
        content=(f"A human reviewer REJECTED the call to '{name}'. Do not retry it. "
                 f"Either answer the user without it, or explain you cannot proceed."),
        tool_call_id=tool_call["id"],
    )
    return Command(goto="agent", update={"messages": [rejection]})


def should_continue(state: AgentState) -> str:
    """Router after the agent speaks: tool call -> approval gate; else -> END."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        log("ROUTER", "agent requested a tool → approval gate", C.BLUE)
        return "human_approval"
    log("ROUTER", "agent gave a final answer → ending run", C.GREEN)
    return END


# =============================================================================
# SECTION 9 — WIRING & DRIVER
# =============================================================================
#         START ─▶ agent ─(tool?)─▶ human_approval ─(ok)─▶ tools ─▶ agent ─▶ …
#                    │  └─(no)─▶ END          └─(reject)─▶ agent
# -----------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["human_approval", END])
    builder.add_edge("tools", "agent")   # after a tool runs, think again

    # Checkpointer is REQUIRED for interrupt()/resume to work (it stores the
    # frozen state so a paused run can be thawed and continued).
    return builder.compile(checkpointer=InMemorySaver())


def run_agent(graph, question):
    """Run one question. Loops only because the graph may PAUSE for approval:
    invoke → finished? print answer : show the pause, ask you, resume, repeat."""
    section(f"NEW RUN — {question}")
    config = {"configurable": {"thread_id": str(uuid.uuid4())},
              "recursion_limit": RECURSION_LIMIT}
    graph_input = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        "pending_images": [],
    }

    step = 0
    while True:
        step += 1
        print(f"\n{C.BOLD}{C.GREEN}───── GRAPH TURN {step} ─────{C.OFF}")
        result = graph.invoke(graph_input, config=config)

        if "__interrupt__" not in result:
            final = result["messages"][-1].content
            section("FINAL ANSWER")
            print(f"{C.BOLD}{C.GREEN}{final}{C.OFF}\n")
            return final

        payload = result["__interrupt__"][0].value
        section("⏸  HUMAN APPROVAL REQUIRED")
        log("REVIEW", payload["question"], C.YELLOW)
        log("REVIEW", f"tool: {payload['tool']}", C.YELLOW)
        log_block("ARGS", payload["args"], C.YELLOW)
        answer = input(f"{C.BOLD}{C.MAGENTA}   Approve? [y/N]: {C.OFF}").strip()
        graph_input = Command(resume=answer)   # feed your answer back in, continue


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    section("SETUP")
    log("CONFIG", f"llm={LLM_MODEL} (vision) via OpenRouter", C.CYAN)
    log("CONFIG", f"tools: {[t.name for t in TOOLS]}", C.CYAN)
    log("CONFIG", f"needing approval: {sorted(TOOLS_NEEDING_APPROVAL)}", C.CYAN)
    log("CONFIG", "Qdrant must be running for retrieve_from_document to work.", C.CYAN)

    agent = build_graph()
    log("CONFIG", "graph compiled: agent → human_approval → tools → agent", C.CYAN)

    # RUN 1 — document question: should retrieve chunks + SEE images, then answer.
    run_agent(agent, "According to the attached document, what does the main diagram explain?")

    # RUN 2 — mixed: retrieve from the doc, then web_search for outside context.
    run_agent(agent, "Explain the key concept in the document, and add how it's used in industry today.")

    # RUN 3 — writer tool: gather material, then ask to SAVE a study note (PAUSES).
    run_agent(agent, "Write me a study summary of what the document covers so I can revise it later.")

    section("DONE")
    log("CONFIG", f"any study notes are in ./{STUDY_NOTES_DIR}/", C.CYAN)
