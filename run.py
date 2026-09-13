"""Interactive and CLI runner for customer-support-agent workflow."""

import asyncio
import os
import sys
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

# Ensure customer_support_agent package can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

# Load environment variables from .env
load_dotenv(os.path.join(current_dir, ".env"))
load_dotenv()

from customer_support_agent.agent import root_agent


async def run_query(runner: InMemoryRunner, session_id: str, query: str):
  """Runs a single query through the customer support workflow."""
  print(f"\n[Customer]: {query}")
  message = types.Content(
      role="user",
      parts=[types.Part.from_text(text=query)],
  )

  agent_response_parts = []
  async for event in runner.run_async(
      user_id="user_1",
      session_id=session_id,
      new_message=message,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          agent_response_parts.append(part.text)

  if agent_response_parts:
    print(f"[SwiftShip Agent]:\n{''.join(agent_response_parts)}\n")
  else:
    print("[SwiftShip Agent]: (No message output)\n")


async def main():
  app_name = "customer_support_app"
  runner = InMemoryRunner(agent=root_agent, app_name=app_name)
  session = await runner.session_service.create_session(
      app_name=app_name, user_id="user_1"
  )

  if len(sys.argv) > 1:
    # Single-query execution from command line arguments
    user_query = " ".join(sys.argv[1:])
    await run_query(runner, session.id, user_query)
  else:
    # Interactive REPL mode
    print("=" * 60)
    print("SwiftShip Customer Support Assistant (ADK 2.0)")
    print("Ask a shipping question or type 'exit' / 'quit' to end.")
    print("=" * 60)

    # If running non-interactively or stdin is closed, run sample queries
    if not sys.stdin.isatty():
      sample_queries = [
          "What are your standard and overnight shipping rates?",
          "How can I track package SW-987654321?",
          "How do I return a damaged item?",
          "Can you help me solve this algebra equation: 3x + 5 = 20?",
      ]
      for query in sample_queries:
        await run_query(runner, session.id, query)
      return

    while True:
      try:
        user_input = input("Customer: ").strip()
      except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

      if not user_input:
        continue
      if user_input.lower() in ("exit", "quit", "q"):
        print("Goodbye!")
        break

      await run_query(runner, session.id, user_input)


if __name__ == "__main__":
  asyncio.run(main())
