"""Core search-agent logic shared by the CLI (main.py) and the desktop GUI (app.py)."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load GOOGLE_API_KEY (and any other vars) from a local .env file
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

#: Results requested per search. Enough for the model to cross-check a claim
#: without spending the context window on a long tail of near-duplicates.
SEARCH_RESULTS = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the search tool to find "
    "real-time web information when needed. When you use search results, "
    "cite the source URLs you relied on."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when no Gemini API key is configured.

    The message is written to be shown to a user as-is: this is by far the most
    common first-run problem, and the underlying library answers it with a
    pydantic validation traceback.
    """


@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return the top results.

    Use this for anything current, factual, or outside your training data.
    """
    # Imported lazily so this module still loads (and the tests still run)
    # where the search backend is unavailable.
    from ddgs import DDGS

    try:
        results = list(DDGS().text(query, max_results=SEARCH_RESULTS))
    except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
        # Returned as text so the agent can decide what to do — answer from its
        # own knowledge, retry, or tell the user. Raising would abort the whole
        # run over one flaky search.
        return f"Search failed: {exc}"

    if not results:
        return f"No results found for {query!r}."

    return "\n\n".join(
        f"{item.get('title', 'Untitled')}\n{item.get('body', '')}\n"
        f"Source: {item.get('href', '')}"
        for item in results
    )


def _require_api_key():
    """Return the configured Gemini key, or raise a message worth showing."""
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "No Gemini API key found.\n\n"
            "Create a file named .env next to this script containing:\n"
            "    GOOGLE_API_KEY=your-key-here\n\n"
            "Get a key at https://aistudio.google.com/apikey"
        )
    return key


_agent = None


def get_agent():
    """Build the agent on first use and reuse it afterwards.

    Deliberately lazy. Building it at import time meant that importing this
    module without a key raised a pydantic ValidationError, so the GUI died
    before its window appeared and neither entrypoint could say anything more
    useful than a traceback. It also made the module impossible to import in a
    test.
    """
    global _agent
    if _agent is None:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=0,
            google_api_key=_require_api_key(),
        )
        _agent = create_agent(llm, [web_search], system_prompt=SYSTEM_PROMPT)
    return _agent


def extract_text(content):
    """Flatten a message's content into plain text.

    Gemini returns content as a list of blocks (e.g. [{"type": "text", ...}])
    rather than a plain string, so handle both shapes.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _sum_usage(messages):
    """Add up token usage across every model message in an agent run.

    An agent turn can involve several model calls (one per tool round-trip),
    so we sum ``usage_metadata`` from all of them for the true cost.
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None)
        if not meta:
            continue
        usage["input_tokens"] += meta.get("input_tokens", 0)
        usage["output_tokens"] += meta.get("output_tokens", 0)
        usage["total_tokens"] += meta.get("total_tokens", 0)
    return usage


def ask(messages):
    """Run the agent over a conversation and return ``(reply_text, usage)``.

    ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``
    dicts. Passing the whole history each call gives the agent memory of the
    conversation so far. ``usage`` is a dict of input/output/total token counts.

    Raises:
        MissingAPIKeyError: If no Gemini API key is configured.
    """
    response = get_agent().invoke({"messages": messages})
    text = extract_text(response["messages"][-1].content)
    usage = _sum_usage(response["messages"])
    return text, usage
