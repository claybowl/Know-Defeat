#!/bin/bash
# Script to test database connection

# Exit on error
set -e

# Configuration
DB_USER="postgres"
DB_NAME="tick_data"
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
CLOUD_SQL_INSTANCE="trading-db"
CLOUD_SQL_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

# Color constants
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print with color
print_status() {
  echo -e "${2}${1}${NC}"
}

# Print step info
print_step() {
  echo -e "\n${YELLOW}== $1 ==${NC}"
}

print_step "Checking Cloud SQL instance"
# Verify that the Cloud SQL instance exists
if gcloud sql instances describe ${CLOUD_SQL_INSTANCE} --project=${PROJECT_ID} &>/dev/null; then
  print_status "Cloud SQL instance '${CLOUD_SQL_INSTANCE}' exists!" "${GREEN}"
else
  print_status "Error: Cloud SQL instance '${CLOUD_SQL_INSTANCE}' not found." "${RED}"
  echo "Please check the instance name and make sure you have permissions to access it."
  echo "You can list available instances with: gcloud sql instances list"
  exit 1
fi

# Ask for database password
echo -n "Enter your database password (input will be hidden): "
read -s DB_PASSWORD
echo ""

print_status "Testing connection to ${CLOUD_SQL_CONNECTION_NAME}..." "${YELLOW}"

# Option 1: Use Cloud SQL Proxy (for local testing)
if [ "$1" = "proxy" ]; then
  echo "Testing connection using Cloud SQL Proxy..."
  
  # Check if Cloud SQL Proxy is installed
  if ! command -v cloud_sql_proxy &> /dev/null; then
    echo "Cloud SQL Proxy not found. Installing..."
    
    # Download and install Cloud SQL Proxy
    curl -o cloud_sql_proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.6.0/cloud-sql-proxy.linux.amd64
    chmod +x cloud_sql_proxy
    
    # Start the proxy in background
    echo "Starting Cloud SQL Proxy..."
    ./cloud_sql_proxy "${CLOUD_SQL_CONNECTION_NAME}" &
    PROXY_PID=$!
    
    # Give proxy time to start
    sleep 5
    
    # Set connection to localhost for proxy
    export DB_HOST="127.0.0.1"
    export DB_PORT="5432"
  else
    echo "Starting Cloud SQL Proxy..."
    cloud_sql_proxy "${CLOUD_SQL_CONNECTION_NAME}" &
    PROXY_PID=$!
    
    # Give proxy time to start
    sleep 5
    
    # Set connection to localhost for proxy
    export DB_HOST="127.0.0.1"
    export DB_PORT="5432"
  fi
  
  # Run test with proxy connection
  export DB_PASSWORD="${DB_PASSWORD}"
  export DB_USER="${DB_USER}"
  export DB_NAME="${DB_NAME}"
  
  node db/check-connection.js
  
  # Kill proxy when done
  if [ -n "$PROXY_PID" ]; then
    echo "Stopping Cloud SQL Proxy..."
    kill $PROXY_PID
  fi
  
# Option 2: Simulated Cloud SQL environment
else
  echo "Testing simulated Cloud SQL connection..."
  
  # Simulate Cloud SQL socket directory
  mkdir -p /tmp/cloudsql
  
  # Set environment variables for direct socket connection
  export DB_PASSWORD="${DB_PASSWORD}"
  export DB_USER="${DB_USER}" 
  export DB_NAME="${DB_NAME}"
  export DB_HOST="/cloudsql"
  export CLOUD_SQL_CONNECTION_NAME="${CLOUD_SQL_CONNECTION_NAME}"
  
  # Run the test
  node db/check-connection.js
fi