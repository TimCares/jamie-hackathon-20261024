"""Minimal agent loop: call the model, run any tools it asks for, repeat.

Run:  uv run agent.py "What is 12 * 7?"
Needs ANTHROPIC_API_KEY in the environment or in a .env file.
"""

import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5-5"

client = anthropic.Anthropic()


# --- Tools: add your own here (schema + a function in TOOL_FUNCS) -------------
TOOLS = [
    {
        "name": "multiply",
        "description": "Multiply two numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    }
]

TOOL_FUNCS = {
    "multiply": lambda a, b: a * b,
}


# --- The loop -----------------------------------------------------------------
def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[tool] {block.name}({block.input})")
            try:
                output, is_error = str(TOOL_FUNCS[block.name](**block.input)), False
            except Exception as e:
                output, is_error = f"{type(e).__name__}: {e}", True
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print(run_agent(" ".join(sys.argv[1:]) or "What is 12 * 7?"))
