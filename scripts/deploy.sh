#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

REPO_PATH="/root/Prophet-Forecasting-For-Portfolio-Optimisation"
SERVICE_NAME="streamlit-app.service"

echo "Navigating to repository..."
cd $REPO_PATH

# 1. Pull the latest code
echo "Pulling latest code from GitHub..."
git pull

# 2. Check for dependency changes and install them
if git diff --name-only HEAD@{1} HEAD | grep -q "pyproject.toml"; then
    echo "pyproject.toml changed. Installing new dependencies..."
    /root/.local/bin/poetry install
fi

# 3. Restart the Streamlit service
echo "Restarting Systemd service: $SERVICE_NAME"
sudo systemctl restart $SERVICE_NAME

echo "Deployment complete."