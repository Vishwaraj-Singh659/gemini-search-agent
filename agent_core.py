"""Core search-agent logic shared by the CLI (main.py) and the desktop GUI (app.py)."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load GOOGLE_API_KEY (and any other vars) from a local .env file
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

# 1. Initialize the LLM via Google Gemini
# Gemini 2.5 Flash is fast and highly capable of tool calling and reasoning.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# 2. Initialize the Free Search Tool (No API Key required)
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# 3. Assemble the Agent (langgraph-based, the v1 way)
agent = create_agent(
    llm,
    tools,
    system_prompt=(
        "You are a helpful assistant. Use the search tool to find "
        "real-time web information when needed."
    ),
)


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
    """
    response = agent.invoke({"messages": messages})
    text = extract_text(response["messages"][-1].content)
    usage = _sum_usage(response["messages"])
    return text, usage
