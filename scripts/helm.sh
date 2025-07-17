#!/bin/bash

set -ex

# This script deploys the slm-server Helm chart.
# It sets the image name, tag, and persistence hostPath.

# --- Configuration ---
# PLEASE REPLACE THESE PLACEHOLDER VALUES
RELEASE_NAME="slm-server"
NAMESPACE="default"
IMAGE_NAME="your-docker-registry/slm-server" # e.g., docker.io/xinyu/slm-server
IMAGE_TAG="latest"                          # e.g., v1.2.3
NODE_NAME=""                                # The name of the k8s node, e.g., "gke-main-pool-12345"

# --- Script Logic ---
# Get the absolute path of the script's directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Get the project root directory (one level up from scripts/)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
MODELS_PATH="$PROJECT_ROOT/models"

# Check if NODE_NAME is set
if [ -z "$NODE_NAME" ]; then
    echo "Error: NODE_NAME is not set. Please edit scripts/helm.sh and specify the target node name."
    exit 1
fi

# Deploy using Helm
# - `upgrade --install`: Installs the chart if the release doesn't exist, or upgrades it if it does.
# - `--namespace`: Specifies the namespace to deploy into.
# - `--create-namespace`: Creates the namespace if it doesn't exist.
# - `--set`: Overrides values in the values.yaml file.
helm upgrade --install $RELEASE_NAME \
  --namespace $NAMESPACE \
  --create-namespace \
  --set image.repository=$IMAGE_NAME \
  --set image.tag=$IMAGE_TAG \
  --set persistence.hostPath=$MODELS_PATH \
  --set persistence.nodeName=$NODE_NAME \
  $PROJECT_ROOT/deploy/helm

echo "Deployment script finished."
