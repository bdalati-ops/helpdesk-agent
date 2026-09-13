"""FastAPI web server serving the Customer Support Agent and interactive Web UI."""

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.adk.runners import InMemoryRunner
from google.genai import types

# Setup paths and load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
  sys.path.insert(0, parent_dir)

load_dotenv(os.path.join(current_dir, ".env"))
load_dotenv()

from customer_support_agent.agent import root_agent

app = FastAPI(
    title="SwiftShip Customer Support Agent",
    description="ADK 2.0 Graph Workflow Agent for Shipping Customer Support",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static directory for web interface
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize ADK Runner and Session Store
APP_NAME = "swiftship_customer_support"
runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
sessions: dict[str, str] = {}


class ChatRequest(BaseModel):
  message: str
  session_id: Optional[str] = None


class ChatResponse(BaseModel):
  response: str
  session_id: str
  nodes_visited: list[str]
  route: Optional[str] = None
  reasoning: Optional[str] = None


@app.get("/health")
async def health_check():
  """Health check endpoint for Cloud Run."""
  return {"status": "ok", "app": APP_NAME, "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
  """Serves the main customer support chat interface."""
  index_path = os.path.join(static_dir, "index.html")
  if os.path.exists(index_path):
    return FileResponse(index_path)
  return HTMLResponse("<h1>SwiftShip Customer Support</h1><p>UI loading...</p>")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  """Processes a user message through the ADK 2.0 graph workflow."""
  user_text = request.message.strip()
  if not user_text:
    raise HTTPException(status_code=400, detail="Message cannot be empty.")

  # Ensure or retrieve session
  session_id = request.session_id
  if not session_id or session_id not in sessions:
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id="web_user",
    )
    session_id = session.id
    sessions[session_id] = session_id

  message_content = types.Content(
      role="user",
      parts=[types.Part.from_text(text=user_text)],
  )

  nodes_visited = []
  response_texts = []
  route_taken = None
  reasoning_text = None

  try:
    async for event in runner.run_async(
        user_id="web_user",
        session_id=session_id,
        new_message=message_content,
    ):
      # Track graph execution nodes
      if event.node_info and event.node_info.path:
        node_name = event.node_info.path.split("/")[-1].split("@")[0]
        if node_name not in nodes_visited:
          nodes_visited.append(node_name)

      # Track route if emitted
      if hasattr(event, "actions") and event.actions and event.actions.route:
        route_taken = event.actions.route

      # Collect agent messages
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            # Check if this part contains classifier JSON metadata
            if '"category":' in part.text and '"reasoning":' in part.text:
              try:
                import json
                parsed = json.loads(part.text)
                route_taken = parsed.get("category", route_taken)
                reasoning_text = parsed.get("reasoning")
              except Exception:
                pass
            else:
              response_texts.append(part.text)

    # Determine fallback route if not explicitly captured
    if not route_taken:
      if "shipping_faq_agent" in nodes_visited:
        route_taken = "shipping"
      elif "decline_unrelated_query" in nodes_visited:
        route_taken = "unrelated"

    final_response = "".join(response_texts).strip()
    if not final_response:
      final_response = "Thank you for contacting SwiftShip. Your request has been processed."

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        nodes_visited=nodes_visited,
        route=route_taken,
        reasoning=reasoning_text,
    )

  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Workflow execution error: {str(e)}"
    )


if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8080))
  uvicorn.run(app, host="0.0.0.0", port=port)
