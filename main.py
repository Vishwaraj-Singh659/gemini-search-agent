import sys
import time

from agent_core import MissingAPIKeyError, ask

# Run the Agent as a simple CLI chat loop.
# For the desktop GUI version, run:  python app.py
if __name__ == "__main__":
    # Fail before the prompt rather than on the first query, so a missing key
    # is one clear message instead of a traceback after the user has typed.
    try:
        from agent_core import get_agent

        get_agent()
    except MissingAPIKeyError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    history = []  # accumulated {"role", "content"} turns for conversation memory
    while True:
        query = input("\nEnter your query or 'exit'/'quit' to stop: ")
        if query.lower() in {"exit", "quit"}:
            print("\nExiting.......\n")
            time.sleep(2)
            break

        history.append({"role": "user", "content": query})
        try:
            answer, usage = ask(history)
            history.append({"role": "assistant", "content": answer})
            print("\nFinal Answer:")
            print(answer)
            print(
                f"\n[tokens] input: {usage['input_tokens']}  "
                f"output: {usage['output_tokens']}  total: {usage['total_tokens']}"
            )
        except Exception as e:
            print(f"An error occurred: {e}")
