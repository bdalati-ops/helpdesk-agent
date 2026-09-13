variable "project_id" {
  description = "The Google Cloud Project ID where infrastructure will be provisioned."
  type        = string
}

variable "region" {
  description = "The Google Cloud region for deploying resources."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run service."
  type        = string
  default     = "customer-support-agent"
}

variable "gemini_api_key_secret_id" {
  description = "The Secret Manager Secret ID for the Gemini API key."
  type        = string
  default     = "gemini-api-key"
}

variable "container_image" {
  description = "The container image URI to deploy to Cloud Run."
  type        = string
  default     = "us-central1-docker.pkg.dev/PROJECT_ID/customer-support-repo/agent:latest"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances (0 enables scale-to-zero)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances."
  type        = number
  default     = 5
}

variable "cpu_limit" {
  description = "CPU allocated per container instance."
  type        = string
  default     = "1"
}

variable "memory_limit" {
  description = "Memory allocated per container instance."
  type        = string
  default     = "1Gi"
}

variable "allow_unauthenticated" {
  description = "Whether to allow public unauthenticated invocations of the Cloud Run service."
  type        = bool
  default     = true
}
