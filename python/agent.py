"""Minimal LangGraph agent loop: LLM -> (tool calls -> LLM)* -> answer.

Run:  uv run agent.py "What is 12 * 7?"
Needs ANTHROPIC_API_KEY in the environment or in a .env file.
"""

import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

MODEL = "claude-sonnet-5-5"


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


TOOLS = [multiply]


llm = ChatAnthropic(model=MODEL).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def route(state: MessagesState) -> str:
    """Loop to the tools node while the model keeps requesting tool calls."""
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route, ["tools", END])
builder.add_edge("tools", "model")

graph = builder.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is 12 * 7?"
    result = graph.invoke({"messages": [("user", question)]})
    for message in result["messages"]:
        message.pretty_print()
