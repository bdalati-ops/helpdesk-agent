#!/usr/bin/env bash
# Deploy SwiftShip Customer Support Agent to Google Cloud Run
set -euo pipefail

PROJECT_ID="${1:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-customer-support-agent}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is required. Usage: ./deploy.sh <PROJECT_ID> [REGION] [SERVICE_NAME]"
  exit 1
fi

# Load local .env if present and GOOGLE_API_KEY is not already in environment
if [[ -z "${GOOGLE_API_KEY:-}" ]] && [[ -f ".env" ]]; then
  export $(grep -v '^#' .env | xargs) || true
fi

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "Error: GOOGLE_API_KEY environment variable is required."
  exit 1
fi

echo "============================================================"
echo "Deploying SwiftShip Customer Support Agent to Cloud Run"
echo "Project:      ${PROJECT_ID}"
echo "Region:       ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "============================================================"

# Ensure gcloud is configured with the project
echo "Checking project configuration..."
gcloud config set project "${PROJECT_ID}" || true

# Deploy source directly via Cloud Run Buildpacks / Dockerfile
echo "Submitting deployment to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_GENAI_USE_ENTERPRISE=FALSE,GOOGLE_API_KEY=${GOOGLE_API_KEY}" \
  --min-instances 0 \
  --max-instances 5 \
  --memory 1Gi \
  --cpu 1

echo ""
echo "Deployment completed successfully!"
echo "Retrieve URL with: gcloud run services describe ${SERVICE_NAME} --project ${PROJECT_ID} --region ${REGION} --format='value(status.url)'"
