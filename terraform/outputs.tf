output "service_url" {
  description = "The publicly accessible URL of the deployed Cloud Run customer support service."
  value       = google_cloud_run_v2_service.agent_service.uri
}

output "artifact_registry_repo" {
  description = "Artifact Registry Docker repository path."
  value       = "${google_artifact_registry_repository.agent_repo.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.agent_repo.repository_id}"
}

output "service_account_email" {
  description = "Email of the dedicated Cloud Run runtime service account."
  value       = google_service_account.cloud_run_sa.email
}

output "secret_manager_secret_id" {
  description = "Secret ID created in Google Cloud Secret Manager."
  value       = google_secret_manager_secret.gemini_api_key.secret_id
}
