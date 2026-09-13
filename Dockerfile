# Dockerfile for SwiftShip Customer Support Agent on Cloud Run
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-dotenv

# Copy application files and static UI assets
COPY . /app/customer_support_agent
COPY . /app

# Expose container port
EXPOSE 8080

# Start FastAPI server with Uvicorn
CMD exec uvicorn customer_support_agent.server:app --host 0.0.0.0 --port ${PORT}
