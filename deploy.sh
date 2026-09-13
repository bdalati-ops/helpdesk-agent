#!/usr/bin/env bash
# Deploy SwiftShip Customer Support Agent to Google Cloud Run with Secret Manager
set -euo pipefail

PROJECT_ID="${1:-${GOOGLE_CLOUD_PROJECT:-}}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-customer-support-agent}"
SECRET_NAME="${SECRET_NAME:-gemini-api-key}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is required. Usage: ./deploy.sh <PROJECT_ID> [REGION] [SERVICE_NAME]"
  exit 1
fi

echo "============================================================"
echo "Deploying SwiftShip Customer Support Agent to Cloud Run"
echo "Project:      ${PROJECT_ID}"
echo "Region:       ${REGION}"
echo "Service Name: ${SERVICE_NAME}"
echo "Secret Name:  ${SECRET_NAME}"
echo "============================================================"

# Ensure gcloud is configured with the target project
echo "Configuring target project..."
gcloud config set project "${PROJECT_ID}" || true

# Enable required Google Cloud APIs
echo "Ensuring required GCP APIs are enabled..."
gcloud services enable \
  run.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT_ID}" || true

# Check or configure Secret Manager secret if GOOGLE_API_KEY is available locally
if [[ -n "${GOOGLE_API_KEY:-}" ]] || [[ -f ".env" ]]; then
  if [[ -z "${GOOGLE_API_KEY:-}" ]] && [[ -f ".env" ]]; then
    export $(grep -E '^GOOGLE_API_KEY=' .env | xargs) || true
  fi

  if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    if ! gcloud secrets describe "${SECRET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
      echo "Creating secret '${SECRET_NAME}' in Secret Manager..."
      echo -n "${GOOGLE_API_KEY}" | gcloud secrets create "${SECRET_NAME}" \
        --replication-policy="automatic" \
        --data-file=- \
        --project "${PROJECT_ID}" || true
    fi
  fi
fi

# Determine whether secret exists in Secret Manager
SECRETS_FLAG=""
ENV_VARS_FLAG="GOOGLE_GENAI_USE_ENTERPRISE=FALSE,GEMINI_API_KEY_SECRET_ID=${SECRET_NAME}"

if gcloud secrets describe "${SECRET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "✓ Found secret '${SECRET_NAME}' in Secret Manager. Mounting via Cloud Run secrets..."
  SECRETS_FLAG="--set-secrets=GOOGLE_API_KEY=${SECRET_NAME}:latest,GEMINI_API_KEY=${SECRET_NAME}:latest"
else
  echo "⚠️ Secret '${SECRET_NAME}' not found in Secret Manager; checking environment variable fallback..."
  if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    ENV_VARS_FLAG="${ENV_VARS_FLAG},GOOGLE_API_KEY=${GOOGLE_API_KEY}"
  else
    echo "Warning: No secret found in Secret Manager and no GOOGLE_API_KEY in environment."
  fi
fi

# Deploy source directly via Cloud Run Buildpacks / Dockerfile
echo "Submitting deployment to Cloud Run..."
DEPLOY_CMD=(
  gcloud run deploy "${SERVICE_NAME}"
  --source .
  --project "${PROJECT_ID}"
  --region "${REGION}"
  --port 8080
  --allow-unauthenticated
  --set-env-vars "${ENV_VARS_FLAG}"
  --min-instances 0
  --max-instances 5
  --memory 1Gi
  --cpu 1
)

if [[ -n "${SECRETS_FLAG}" ]]; then
  DEPLOY_CMD+=("${SECRETS_FLAG}")
fi

"${DEPLOY_CMD[@]}"

echo ""
echo "============================================================"
echo "Deployment completed successfully!"
echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)'
echo "============================================================"
