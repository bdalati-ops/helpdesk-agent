# ==============================================================================
# Google Cloud APIs Enablement
# ==============================================================================
resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com"
  ])

  service                    = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}

# ==============================================================================
# Artifact Registry Repository for Container Images
# ==============================================================================
resource "google_artifact_registry_repository" "agent_repo" {
  provider      = google
  location      = var.region
  repository_id = "customer-support-repo"
  description   = "Docker repository for SwiftShip Customer Support Agent"
  format        = "DOCKER"

  depends_on = [google_project_service.enabled_apis]
}

# ==============================================================================
# Secret Manager: Gemini API Key
# ==============================================================================
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = var.gemini_api_key_secret_id

  replication {
    auto {}
  }

  labels = {
    app         = "swiftship"
    managed_by  = "terraform"
    environment = "production"
  }

  depends_on = [google_project_service.enabled_apis]
}

# ==============================================================================
# Dedicated Service Account for Cloud Run (Least Privilege)
# ==============================================================================
resource "google_service_account" "cloud_run_sa" {
  account_id   = "sa-customer-support-agent"
  display_name = "SwiftShip Customer Support Agent Cloud Run Identity"
  description  = "Service account runtime identity for SwiftShip customer support service"
}

# Grant Secret Accessor only to the Gemini secret
resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Grant Cloud Logging and Tracing permissions
resource "google_project_iam_member" "log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

resource "google_project_iam_member" "trace_agent" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# ==============================================================================
# Cloud Run v2 Service Deployment with Secret Manager Mounts
# ==============================================================================
resource "google_cloud_run_v2_service" "agent_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.cloud_run_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }

      env {
        name  = "GOOGLE_GENAI_USE_ENTERPRISE"
        value = "FALSE"
      }

      env {
        name  = "GEMINI_API_KEY_SECRET_ID"
        value = var.gemini_api_key_secret_id
      }

      # Securely mount Gemini API Key from Secret Manager
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        timeout_seconds   = 10
        period_seconds    = 5
        failure_threshold = 3
        tcp_socket {
          port = 8080
        }
      }

      liveness_probe {
        timeout_seconds   = 10
        period_seconds    = 15
        failure_threshold = 3
        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_secret_manager_secret_iam_member.secret_accessor,
    google_project_service.enabled_apis
  ]
}

# ==============================================================================
# Public Access Policy (Conditional)
# ==============================================================================
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count = var.allow_unauthenticated ? 1 : 0

  name     = google_cloud_run_v2_service.agent_service.name
  location = google_cloud_run_v2_service.agent_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
