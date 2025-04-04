#!/bin/bash
# Setup Google Cloud SQL Proxy for local development

# Check if the Cloud SQL Proxy is already installed
if [ ! -f ~/cloud-sql-proxy ]; then
    echo "Downloading Cloud SQL Proxy..."
    curl -o ~/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.0.0/cloud-sql-proxy.linux.amd64
    chmod +x ~/cloud-sql-proxy
else
    echo "Cloud SQL Proxy already installed."
fi

# Make sure gcloud is authenticated
echo "Checking gcloud authentication..."
gcloud auth list

# Get the current project
PROJECT_ID=$(gcloud config get-value project)
echo "Current project: $PROJECT_ID"

# Start the Cloud SQL Proxy
echo "Starting Cloud SQL Proxy for know-defeat-trading:us-central1:trading-db"
echo "This will create a connection to your Cloud SQL instance on localhost:5432"
echo "Press Ctrl+C to stop the proxy when done."

# Start the proxy
~/cloud-sql-proxy --port 5432 know-defeat-trading:us-central1:trading-db