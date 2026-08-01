"""
A minimal ReAct agent, written from scratch.

No LangChain, no LangGraph, no native function calling. Just:
  - Gemini for the LLM
  - Tavily for web search
  - Chroma + Gemini embeddings for long-term memory
  - A while loop, a regex, and a stop sequence

The ReAct pattern looks like this:

    Question: what the user asked
    Thought:  the model reasons about what to do
    Action:   the tool it wants to call
    Action Input: the argument for that tool
    Observation: <-- WE write this, not the model. It's the real tool output.
    Thought:  ...the model reasons again, now knowing the observation
    ...repeat...
    Final Answer: the answer for the user

Everything is logged so you can watch each step happen.

Run it:  python "learningçagentic_reAct.py"
"""

import os
import re
import uuid
import textwrap
from datetime import datetime, timezone

import requests
import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# =============================================================================
# SECTION 1 — CONFIG & API KEYS
# =============================================================================
#
#   >>> THIS IS WHERE THE API KEYS GO. <<<
#
# We route through OpenRouter (https://openrouter.ai) instead of calling Gemini
# directly. OpenRouter exposes an OpenAI-compatible API that proxies to many
# providers — including Google's Gemini models for both chat AND embeddings —
# under one key. Handy if your own Google Cloud project gets rejected (403s).
#
# Create a file called  .env  next to this script (it is already in .gitignore,
# so it will never be committed) containing exactly two lines:
#
#     GEMINI_API_KEY=sk-or-v1-...your OpenRouter key here...
#     TAVILY_API_KEY=tvly-...your key here...
#
# (The variable is still named GEMINI_API_KEY even though the value is an
# OpenRouter key — OpenRouter is just the pipe we're pointing at Gemini through.)
#
# Get them here:
#     OpenRouter -> https://openrouter.ai/keys
#     Tavily     -> https://app.tavily.com/home
#
# load_dotenv() reads that file and puts the values into os.environ, so the
# keys never appear in your source code.
# -----------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Fail loudly and immediately, instead of mysteriously halfway through a run.
if not OPENROUTER_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Put it in a .env file (see SECTION 1).")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set. Put it in a .env file (see SECTION 1).")

# Which models we use (OpenRouter model slugs — provider/model-name).
LLM_MODEL = "google/gemini-2.5-flash"          # the reasoning brain
EMBED_MODEL = "google/gemini-embedding-001"    # turns text into vectors for memory

# Where Chroma writes its files, and the safety cap on the agent loop.
CHROMA_PATH = "./agent_memory_db"
MEMORY_COLLECTION = "long_term_memory"
MAX_ITERATIONS = 6                      # stops a broken agent from looping forever
SEARCH_RESULT_CHAR_LIMIT = 500          # truncate each web result so context stays small


# =============================================================================
# SECTION 2 — LOGGING HELPERS
# =============================================================================
# The whole point of this exercise is *seeing* the loop, so we log aggressively.
# These are just print() with ANSI colors and a bit of formatting.
# -----------------------------------------------------------------------------

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
    """Print a big banner. Used to mark the boundaries between the 8 sections."""
    print(f"\n{C.BOLD}{C.BLUE}{'=' * 70}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}  {title}{C.OFF}")
    print(f"{C.BOLD}{C.BLUE}{'=' * 70}{C.OFF}")


def log(tag, message, color=C.GREY):
    """Print one labelled line, indenting any wrapped continuation lines."""
    body = textwrap.indent(str(message), " " * 14).strip()
    print(f"{color}{C.BOLD}[{tag:^10}]{C.OFF} {color}{body}{C.OFF}")


def log_block(tag, message, color=C.GREY):
    """Print a multi-line block (raw LLM output, tool output, etc.) inside a frame."""
    print(f"{color}{C.BOLD}[{tag:^10}]{C.OFF} {color}┌{'─' * 60}{C.OFF}")
    for line in str(message).splitlines() or [""]:
        print(f"{color}             │ {line}{C.OFF}")
    print(f"{color}             └{'─' * 60}{C.OFF}")


# =============================================================================
# SECTION 3 — THE LLM CALL (Gemini, via OpenRouter's OpenAI-compatible API)
# =============================================================================
# One function: give it a prompt string, get back the model's raw text.
#
# The single most important line in this section is:
#
#       stop=["Observation:"]
#
# Why it matters: the model has seen thousands of ReAct traces in training, so
# after writing "Action Input: ..." it will happily keep going and *invent* an
# Observation out of thin air — a hallucinated tool result. The stop sequence
# makes the model physically stop generating the instant it produces that
# token. So the model literally cannot fake a tool result. We then run the
# real tool and paste the real Observation in ourselves.
#
# Because we're going through OpenRouter, the request/response shape is the
# standard OpenAI chat-completions format (messages list with role/content),
# not Gemini's native SDK shape — that's the whole point of OpenRouter: one
# API surface in front of many providers.
# -----------------------------------------------------------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def call_llm(prompt, system_prompt):
    """Send the prompt to Gemini (through OpenRouter) and return the raw reply text."""
    log("LLM CALL", f"→ sending {len(prompt)} chars to {LLM_MODEL}", C.CYAN)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,               # deterministic: we want reproducible traces
        stop=["Observation:"],         # <-- the model can never fake a tool result
        max_tokens=1024,               # one ReAct step is short; also avoids 402s from
                                        # OpenRouter defaulting to the model's max (65535)
    )

    text = (response.choices[0].message.content or "").strip()
    log("LLM CALL", f"← received {len(text)} chars", C.CYAN)
    log_block("RAW LLM", text, C.GREY)
    return text


# =============================================================================
# SECTION 4 — LONG-TERM MEMORY (Chroma + Gemini embeddings)
# =============================================================================
# A "vector database" is much less mystical than it sounds:
#
#   1. Turn a piece of text into a list of numbers (an embedding) that captures
#      its meaning. Similar meanings -> similar numbers.
#   2. Store those numbers alongside the original text and some metadata.
#   3. To search: embed the *query* the same way, then find the stored vectors
#      that are geometrically closest to it (cosine similarity).
#
# That's it. Chroma just handles the storage and the nearest-neighbour math.
# We call Gemini's embedding model ourselves and hand Chroma the raw vectors,
# rather than letting it hide that step, so you can see it happening.
# -----------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

memory_collection = chroma_client.get_or_create_collection(
    name=MEMORY_COLLECTION,
    metadata={"hnsw:space": "cosine"},  # measure distance by cosine similarity
)


def embed(text, task_type):
    """
    Turn text into a vector using Gemini's embedding model, via OpenRouter.

    Note: called natively, Gemini's embedding API accepts a `task_type` hint
    (RETRIEVAL_DOCUMENT for text being stored vs. RETRIEVAL_QUERY for a
    question being asked) and a chosen output size — using the matching type on
    each side improves retrieval. OpenRouter's generic /embeddings endpoint is
    OpenAI-shaped and doesn't expose either of those Gemini-specific knobs, so
    we can't pass task_type through here; we keep the parameter (and log it)
    only so the calling code still documents *which* side of the search each
    call belongs to. The vector's dimensionality is whatever the model returns
    by default.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vector = response.data[0].embedding
    log("EMBED", f"{task_type}: '{text[:45]}...' → vector of {len(vector)} floats", C.MAGENTA)
    return vector


# =============================================================================
# SECTION 5 — THE TOOLS
# =============================================================================
# Every tool has the SAME shape: it takes one string, returns one string.
#
# That uniformity is the trick that makes the loop in Section 8 generic — it can
# call any tool without knowing anything about it, just  tool_fn(action_input).
# The string it returns becomes the Observation the model sees next.
# -----------------------------------------------------------------------------

def web_search(query):
    """
    TOOL 1: search the web with Tavily.

    We call Tavily's REST endpoint directly with `requests` instead of using
    their SDK — it's a single HTTP POST, and seeing it raw is the point here.
    """
    log("TOOL", f"web_search(query='{query}')", C.YELLOW)

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3,           # keep it small: 3 results is plenty
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()  # turns an HTTP 4xx/5xx into an exception
    data = response.json()

    results = data.get("results", [])
    if not results:
        return "No results found for that query."

    # Format the results into a compact string, TRUNCATING each one. Without
    # this truncation a few searches would flood the context window and the
    # agent would get slow, expensive, and confused.
    lines = []
    for i, r in enumerate(results, start=1):
        content = r.get("content", "")[:SEARCH_RESULT_CHAR_LIMIT]
        lines.append(f"[{i}] {r.get('title', 'untitled')}\n    {content}\n    (source: {r.get('url', '')})")

    output = "\n".join(lines)
    log("TOOL", f"web_search returned {len(results)} results ({len(output)} chars)", C.YELLOW)
    return output


def save_memory(text):
    """
    TOOL 2: store a durable fact in the vector DB.

    Steps: embed the text -> store vector + original text + metadata in Chroma.
    The metadata (timestamp) is just extra info carried alongside the vector;
    it isn't searched over, but you get it back with every retrieved result.
    """
    log("TOOL", f"save_memory(text='{text}')", C.YELLOW)

    vector = embed(text, task_type="RETRIEVAL_DOCUMENT")

    memory_collection.add(
        ids=[str(uuid.uuid4())],                # every record needs a unique id
        embeddings=[vector],                    # the numbers we search over
        documents=[text],                       # the original text we get back
        metadatas=[{                            # anything else we want to keep
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "agent",
        }],
    )

    log("TOOL", f"saved. collection now holds {memory_collection.count()} memories", C.YELLOW)
    return f"Saved to long-term memory: '{text}'"


def search_memory(query):
    """
    TOOL 3: retrieve relevant facts from the vector DB.

    Steps: embed the query -> ask Chroma for the nearest stored vectors ->
    return their original texts. Note we never string-match; two memories can
    share zero words with the query and still be retrieved, because we're
    comparing *meaning* (vectors), not characters.
    """
    log("TOOL", f"search_memory(query='{query}')", C.YELLOW)

    if memory_collection.count() == 0:
        return "Long-term memory is empty. Nothing has been saved yet."

    vector = embed(query, task_type="RETRIEVAL_QUERY")

    results = memory_collection.query(
        query_embeddings=[vector],
        n_results=min(3, memory_collection.count()),  # can't ask for more than we have
    )

    # Chroma returns lists-of-lists (one inner list per query). We sent one
    # query, so we take index [0] of each.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    lines = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # cosine distance: 0 = identical meaning, 2 = opposite. Lower is better.
        similarity = 1 - dist
        lines.append(f"- {doc}  (saved: {meta.get('timestamp', '?')[:10]}, similarity: {similarity:.2f})")
        log("MEMORY", f"hit: '{doc[:40]}...' similarity={similarity:.2f}", C.MAGENTA)

    return "Found these memories:\n" + "\n".join(lines)


# =============================================================================
# SECTION 6 — THE TOOL REGISTRY
# =============================================================================
# This dict IS the "tool binding" that frameworks make into a whole subsystem.
# Name -> function. The loop looks up whatever name the model wrote and calls it.
# -----------------------------------------------------------------------------

TOOLS = {
    "web_search": web_search,
    "save_memory": save_memory,
    "search_memory": search_memory,
}

TOOL_DESCRIPTIONS = """
- web_search: Searches the live internet. Use for current events, facts you don't
  know, or anything that may have changed recently. Input: a search query string.

- save_memory: Saves ONE durable fact to long-term memory so it survives across
  conversations. Input: the fact to remember, written as a complete sentence.

- search_memory: Looks up facts previously saved to long-term memory. Input: a
  description of what you're looking for.
""".strip()


# =============================================================================
# SECTION 7 — THE SYSTEM PROMPT
# =============================================================================
# With no native function calling, the prompt is the *entire* interface between
# the model and our code. It has to do three jobs:
#
#   1. Teach the exact output grammar we're going to parse with a regex.
#   2. Forbid the model from writing "Observation:" itself. (The stop sequence
#      enforces this, but telling it as well makes the model behave better.)
#   3. Tell it WHEN to use memory — otherwise it either never saves anything,
#      or compulsively saves every trivial thing you say.
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a helpful research assistant that reasons step by step.

You have access to these tools:
{TOOL_DESCRIPTIONS}

You must ALWAYS reply using this exact format:

Thought: your reasoning about what to do next
Action: the name of the tool to use, exactly one of [web_search, save_memory, search_memory]
Action Input: the input to give that tool

Then STOP. Do not write anything else.

The system will run the tool for real and reply with a line beginning "Observation:".
NEVER write an Observation yourself. You do not know what the tool returns until the
system tells you. Inventing an observation is a serious error.

After you see the Observation, continue with another Thought, and either another
Action, or — once you can answer the user — finish with:

Thought: I now know the final answer
Final Answer: your complete answer to the user

RULES FOR MEMORY (think about these; do not use memory reflexively):
- Call search_memory FIRST when the question refers to the user personally, to
  things you were told before, or to anything you might already have been told
  ("what do you know about me", "what's my favourite...", "where do I work").
- Call save_memory when the user tells you a durable fact worth remembering
  later: preferences, personal details, decisions, ongoing projects.
- Do NOT save trivia, small talk, or facts you looked up on the web — those can
  always be looked up again.
- Do NOT call a tool at all if you can already answer from the conversation.
  Simple questions ("what is 2+2") should go straight to a Final Answer.

Take exactly ONE action at a time, then wait for the Observation."""


# =============================================================================
# SECTION 8 — THE PARSER AND THE REACT LOOP
# =============================================================================
# This is the heart of the agent. Everything above was setup.
# -----------------------------------------------------------------------------

def parse_llm_output(text):
    """
    Turn the model's free text into something Python can act on.

    Returns one of:
        ("final", "the answer string")
        ("action", ("tool_name", "tool input"))
        ("error",  "what went wrong")

    This is the fragile part of free-text agents, and it's exactly why native
    function calling exists — with function calling the model returns clean JSON
    and none of this regex work is needed. Doing it by hand shows you what that
    convenience is actually buying you.
    """
    # Check for a final answer FIRST. If the model is done, nothing else matters.
    # re.DOTALL lets `.` match newlines, so multi-line answers are captured whole.
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return "final", final_match.group(1).strip()

    # Otherwise look for an Action + Action Input pair.
    action_match = re.search(r"Action:\s*(.*?)\s*\n", text + "\n", re.IGNORECASE)

    # The Action Input runs to the end of the model's output... UNLESS the model
    # disobeyed and wrote its own "Observation:" anyway. The stop sequence should
    # prevent that, but never trust the model: if we didn't cut the match off at
    # Observation, a hallucinated one would get swallowed into the tool input and
    # we'd search the web for "cats\nObservation: cats are fake".
    input_match = re.search(
        r"Action Input:\s*(.*?)(?:\n\s*Observation:|$)", text, re.DOTALL | re.IGNORECASE
    )

    if action_match and input_match:
        tool_name = action_match.group(1).strip()
        # The model sometimes wraps its input in quotes or backticks. Strip them.
        tool_input = input_match.group(1).strip().strip('"').strip("'").strip("`").strip()
        return "action", (tool_name, tool_input)

    # The model produced something we can't use (this does happen — handle it).
    return "error", "Could not find either an 'Action:'/'Action Input:' pair or a 'Final Answer:' in the response."


def execute_tool(tool_name, tool_input):
    """
    Run a real tool and return its output as a string.

    ERROR HANDLING: a tool that raises must NOT crash the agent. We catch the
    exception and turn it into a normal string, which becomes the Observation.
    The model then *reads* the error and can react to it — retry with a different
    query, try another tool, or tell the user it failed. The error becomes part
    of the reasoning instead of ending the program.
    """
    if tool_name not in TOOLS:
        available = ", ".join(TOOLS)
        log("ERROR", f"model asked for unknown tool '{tool_name}'", C.RED)
        return f"Error: no tool named '{tool_name}'. Available tools are: {available}."

    try:
        return TOOLS[tool_name](tool_input)
    except Exception as e:
        # Catching bare Exception is normally bad practice — here it's the point.
        # ANY tool failure has to come back as text the model can reason about.
        log("ERROR", f"tool '{tool_name}' raised {type(e).__name__}: {e}", C.RED)
        return f"Error: the tool '{tool_name}' failed with {type(e).__name__}: {e}. Consider trying a different approach."


def run_agent(question):
    """
    THE REACT LOOP.

    `scratchpad` is the short-term memory: a plain string that grows with every
    step. There is nothing more to it than that. Each iteration we send the model
    the whole question + everything that has happened so far, and it decides what
    to do next. The model itself is stateless; the string is the state.
    """
    section(f"NEW RUN — {question}")

    scratchpad = f"Question: {question}\n"

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n{C.BOLD}{C.GREEN}───── ITERATION {i}/{MAX_ITERATIONS} ─────{C.OFF}")

        # --- STEP 1: ask the model what to do next -------------------------
        # We hand it the entire scratchpad every single time. This is why agents
        # get expensive: the prompt grows with each step of the loop.
        llm_output = call_llm(scratchpad, SYSTEM_PROMPT)

        # --- STEP 2: parse what it said ------------------------------------
        kind, payload = parse_llm_output(llm_output)
        log("PARSE", f"kind={kind}", C.BLUE)

        # --- STEP 3a: it's done ---------------------------------------------
        if kind == "final":
            log("PARSE", "model produced a Final Answer — exiting loop", C.GREEN)
            section("FINAL ANSWER")
            print(f"{C.BOLD}{C.GREEN}{payload}{C.OFF}\n")
            return payload

        # --- STEP 3b: it produced garbage -----------------------------------
        if kind == "error":
            log("PARSE", payload, C.RED)
            # Push the complaint back into the scratchpad and let it try again.
            scratchpad += llm_output + f"\nObservation: {payload} Please follow the required format exactly.\n"
            continue

        # --- STEP 3c: it wants to call a tool -------------------------------
        tool_name, tool_input = payload
        log("PARSE", f"tool={tool_name!r} input={tool_input!r}", C.BLUE)

        observation = execute_tool(tool_name, tool_input)
        log_block("OBSERVATION", observation, C.YELLOW)

        # --- STEP 4: grow the scratchpad and loop ---------------------------
        # This is the actual "agency": we append the model's own words PLUS the
        # real-world result, so on the next pass it reasons with new information
        # it did not have before. Note the trailing newline after the
        # observation — it puts the model's cursor on a fresh line, ready to
        # write its next "Thought:".
        scratchpad += llm_output + f"\nObservation: {observation}\n"

        log("STATE", f"scratchpad is now {len(scratchpad)} chars", C.GREY)

    # --- Safety net ---------------------------------------------------------
    # We fell out of the for-loop, meaning MAX_ITERATIONS was hit without a Final
    # Answer. Without this cap a confused agent would loop (and bill you) forever.
    section("STOPPED — HIT MAX ITERATIONS")
    log("ERROR", f"no final answer after {MAX_ITERATIONS} iterations", C.RED)
    return f"I couldn't reach an answer within {MAX_ITERATIONS} steps."


# =============================================================================
# DEMO — three runs that exercise the three different behaviours
# =============================================================================

if __name__ == "__main__":
    section("SETUP")
    log("CONFIG", f"llm={LLM_MODEL}  embeddings={EMBED_MODEL} (via OpenRouter)", C.CYAN)
    log("CONFIG", f"chroma path={CHROMA_PATH}  memories stored={memory_collection.count()}", C.CYAN)
    log("CONFIG", f"tools registered: {list(TOOLS)}", C.CYAN)

    # RUN 1 — should call save_memory (a durable personal fact).
    run_agent("Remember that I'm building a vision-language model side project, and that I prefer explanations with lots of comments.")

    # RUN 2 — should call search_memory (asks about the user personally).
    run_agent("what is the most lovable languge by people in the world of course after english")

    # RUN 3 — should call web_search (a fact the model can't know).
    run_agent("Who won the mos is the most lovable langiages by people that they want to learn of course after english t recent Formula 1 world championship?")

    section("DONE")
    log("CONFIG", f"long-term memory now holds {memory_collection.count()} facts", C.CYAN)
    log("CONFIG", f"they persist in {CHROMA_PATH}/ — run this again and they'll still be there", C.CYAN)
