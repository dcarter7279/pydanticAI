from pydantic_ai import Agent, ModelRetry
import rizaio
import json


code_agent = Agent("openai:gpt-4o", system_prompt="You are a helpful assistant.")


@code_agent.tool_plain
def execute_code(code: str) -> str:
    """Execute Python code

    Use print() to write the output of your code to stdout.

    Use only the Python standard library and built-in modules. For example, do not use pandas, but you can use csv.

    Use httpx to make http requests.
    """

    print(f"Agent wanted to execute this code:\n```\n{code}\n```")

    riza = rizaio.Riza()
    result = riza.command.exec(
        language="PYTHON", code=code, http={"allow": [{"host": "*"}]}
    )

    if result.exit_code != 0:
        raise ModelRetry(result.stderr)
    if result.stdout == "":
        raise ModelRetry(
            "Code executed successfully but produced no output. "
            "Ensure your code includes print statements to get output."
        )

    print(f"Execution output:\n```\n{result.stdout}\n```")
    return result.stdout


def log_messages(messages):
    """Convert agent messages to JSON-serializable format and save to file."""
    serialized = [
        {
            "role": m.role if hasattr(m, "role") else "unknown",
            "content": m.content if hasattr(m, "content") else str(m),
        }
        for m in messages
    ]
    with open("all_messages.json", "w") as f:
        json.dump(serialized, f, indent=2)


if __name__ == "__main__":
    usr_msg = "Please introduce yourself."
    result = code_agent.run_sync(usr_msg)
    while usr_msg != "quit":
        print(result.data)
        usr_msg = input("> ")
        result = code_agent.run_sync(usr_msg, message_history=result.all_messages())
        log_messages(result.all_messages())
