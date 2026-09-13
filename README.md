# customer-support-agent

An **ADK 2.0 (Agent Development Kit)** graph workflow project implementing an intelligent customer support representative for a shipping and logistics company (**SwiftShip**).

---

## Overview

The workflow acts as an automated triage and customer service pipeline. For every customer query:
1. It records and normalizes the customer's question into session state.
2. An intent classifier (`query_classifier`) evaluates whether the query is **shipping-related** (rates, tracking, delivery status, returns, label generation) or **unrelated** (general trivia, programming questions, weather forecasts, recipes, etc.).
3. Based on the classification, conditional routing branches to:
   - **Shipping FAQ Agent** (`shipping_faq_agent`): A specialized customer service agent that provides comprehensive, friendly answers regarding shipping rates, delivery timelines, tracking status, and return logistics.
   - **Decline Node** (`decline_unrelated_query`): A deterministic node that politely declines to answer non-shipping inquiries and explains the scope of available support.

Deployment files are omitted as requested.

---

## Workflow Topology

```mermaid
flowchart TD
    START([START]) --> ProcessInput["process_user_query\n(Record query in session state)"]
    ProcessInput --> Classifier["query_classifier (Agent)\n(output_schema: QueryClassification)"]
    Classifier --> Router{"route_query\n(Conditional Branch)"}
    Router -->|route: shipping| ShippingFAQ["shipping_faq_agent (Agent)\n(Answers Rates, Tracking, Delivery, Returns)"]
    Router -->|route: unrelated| DeclineNode["decline_unrelated_query (Node)\n(Politely declines to answer)"]
    ShippingFAQ --> END([END])
    DeclineNode --> END([END])
```

---

## Component Breakdown

| Component | Type | Responsibility |
|---|---|---|
| `process_user_query` | Function Node | Takes user message from `START`, outputs the query text, and populates `ctx.state['user_query']` for downstream template resolution. |
| `query_classifier` | `LlmAgent` | Evaluates the query using `gemini-2.5-flash` with a strict structured `output_schema` (`QueryClassification`) returning `"shipping"` or `"unrelated"`. |
| `route_query` | Function Node | Inspects the classification result and yields an `Event(route=...)` directing graph execution. |
| `shipping_faq_agent` | `LlmAgent` | Domain specialist handling shipping rates (ground, priority, overnight, international), tracking milestones, delivery windows/signatures, and return logistics. |
| `decline_unrelated_query` | Function Node | Yields customer-facing polite decline `Event(message=..., output=...)` to handle off-topic requests. |
| `root_agent` | `Workflow` | Top-level graph combining sequential transitions and conditional branching. |

---

## Project Structure

```text
customer-support-agent/
├── .dockerignore          # Docker build exclusions (.env, caches, venv)
├── .env.example           # Environment variable template for Gemini / Vertex AI
├── .gitignore             # Git exclusions for secrets, caches, and virtualenvs
├── Dockerfile             # Production container definition for Cloud Run
├── README.md              # Project documentation
├── __init__.py            # Package entry point exporting agent
├── agent.py               # ADK 2.0 Workflow definitions, schemas, and instrumented nodes
├── deploy.sh              # Sanitized Cloud Run deployment script
├── observability.py       # OpenTelemetry tracing, structured JSON logging, PII redaction & intent tracking
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Python dependencies (google-adk, opentelemetry, etc.)
├── run.py                 # Interactive REPL and CLI runner with telemetry
├── run_mock.py            # Hermetic test runner for DAG verification
├── server.py              # FastAPI server with chat API, tracing middleware, and UI hosting
├── static/
│   └── index.html         # SwiftShip responsive Web UI with DAG trace inspector
└── tests/
    ├── __init__.py
    ├── test_observability.py # Unit tests for tracing, logging, PII redaction, intent tracking
    └── test_workflow.py   # Unit tests for schemas, nodes, and graph structure
```

---

## Observability & Tracing

The agent engine includes an observability framework built on **OpenTelemetry** and structured JSON logging:

### 1. OpenTelemetry Distributed Tracing
- **Span Hierarchy**: Every customer interaction creates a root span (`http.request` or `cli.query_turn`) containing child spans for workflow execution (`workflow.runner_execution`), user query processing (`workflow.process_user_query`), intent routing (`workflow.route_query`), and decline/FAQ execution.
- **Trace Context Propagation**: Automatically sets `X-Trace-Id` on HTTP responses for end-to-end correlation with frontend clients and upstream services.
- **Span Status & Exceptions**: Errors are captured via `span.record_exception()` and marked with `StatusCode.ERROR`.

### 2. Structured JSON Logging with Cloud Logging Correlation
- Every log message is formatted as a single-line JSON object compliant with Google Cloud Logging / W3C standards.
- Automatic correlation with active trace and span IDs:
  - `trace_id` and `span_id`
  - `logging.googleapis.com/trace` (`projects/{PROJECT_ID}/traces/{TRACE_ID}`)
  - `logging.googleapis.com/spanId`
  - `logging.googleapis.com/trace_sampled`
- Contextual attributes and structured error stack traces.

### 3. Intent vs. Outcome Tracking
- Tracks and audits every interaction by comparing:
  - **Detected Intent**: User query intent category (`shipping` vs `unrelated`) and LLM reasoning.
  - **Actual Outcome**: Target node executed (`shipping_faq_agent` vs `decline_unrelated_query`), action taken (`answered_faq` vs `declined_unrelated`), and execution alignment (`aligned` vs `mismatched`).
  - **Performance Metrics**: End-to-end execution latency in milliseconds.
- Emits a dedicated audit log event (`event_type: "intent_vs_outcome"`) and sets span attributes on the active trace.

### 4. PII Redaction Engine
- Real-time sanitization of customer data across logs, telemetry spans, and session state:
  - **Email Addresses**: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` ➔ `[REDACTED_EMAIL]`
  - **Phone Numbers**: 7-digit, 10-digit, and international formats ➔ `[REDACTED_PHONE]`
  - **Credit / Debit Cards**: 13–16 digit payment card numbers ➔ `[REDACTED_CARD]`
  - **Social Security Numbers**: `###-##-####` ➔ `[REDACTED_SSN]`
  - **API Tokens & Secrets**: `ghp_...`, `AIza...`, `ya29...`, `AQ...` ➔ `[REDACTED_SECRET]`
- **Preserved Tracking Numbers**: SwiftShip tracking numbers (`SW-123456789`) are preserved to allow legitimate shipping status lookups.

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Virtual environment (`venv` or `uv`)
- A Google AI Studio API key or GCP Vertex AI project

### 2. Installation

Create and activate a virtual environment, then install dependencies:

```bash
cd customer-support-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Credentials

Copy `.env.example` to `.env` inside the `customer-support-agent/` directory:

```bash
cp .env.example .env
```

Edit `.env` to include your Google Gemini API key:

```bash
GOOGLE_GENAI_USE_ENTERPRISE=FALSE
GOOGLE_API_KEY=AIzaSy...
```

*(Alternatively, if using Vertex AI, configure `GOOGLE_GENAI_USE_ENTERPRISE=TRUE`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION`)*.

---

## Running the Agent

### Using the ADK CLI

**Interactive Terminal Mode:**
```bash
adk run .
```

**Single Query Execution:**
```bash
adk run . "How much does overnight shipping cost for a 5lb box?"
```

**Decline Example:**
```bash
adk run . "Can you help me write a Python script to sort a list?"
```

**Development Web UI:**
```bash
adk web .
```
Then navigate to `http://localhost:8000` to interact via the web interface.

---

### Running Programmatically in Python

You can execute the workflow within Python code using `InMemoryRunner`:

```python
import asyncio
from google.adk.runners import InMemoryRunner
from google.genai import types

from customer_support_agent.agent import root_agent

async def main():
    runner = InMemoryRunner(agent=root_agent, app_name="customer_support")
    session = await runner.session_service.create_session(
        app_name="customer_support",
        user_id="customer_1"
    )

    query = "Where is my package SW-123456789 and when will it arrive?"
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    print(f"Customer: {query}\n")
    async for event in runner.run_async(
        user_id="customer_1",
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent: {part.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Running Tests

Execute the automated test suite with pytest:

```bash
pytest
```
