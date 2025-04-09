#!/bin/bash
# Enhanced deployment with API server and Cloud SQL integration

# Exit on error
set -e

# Color constants for better readability
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

# Configuration
PROJECT_ID="know-defeat-trading"
REGION="us-central1"
SERVICE_NAME="know-defeat-api"  # New name for the API service
DB_USER="postgres"
DB_NAME="tick_data"
CLOUD_SQL_INSTANCE="trading-db"
CLOUD_SQL_CONNECTION_NAME="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

print_step "Setting Google Cloud project"
# Set project ID
gcloud config set project ${PROJECT_ID}

# Check if the Cloud SQL instance exists
if ! gcloud sql instances describe ${CLOUD_SQL_INSTANCE} --project=${PROJECT_ID} &>/dev/null; then
  print_status "Warning: Cloud SQL instance '${CLOUD_SQL_INSTANCE}' not found. You may need to create it first." "${RED}"
  print_status "You can still deploy with mock data, but real database connections will fail." "${YELLOW}"
fi

# Enable necessary APIs
print_step "Enabling required Google Cloud APIs"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com sqladmin.googleapis.com >/dev/null

# If the Cloud SQL instance doesn't exist, offer to create it
if ! gcloud sql instances describe ${CLOUD_SQL_INSTANCE} --project=${PROJECT_ID} &>/dev/null; then
  print_step "Cloud SQL instance setup"
  print_status "The Cloud SQL instance '${CLOUD_SQL_INSTANCE}' doesn't exist yet." "${YELLOW}"
  echo "Do you want to create it now? (yes/no) Note: This will take several minutes."
  read CREATE_SQL_INSTANCE
  
  if [ "$CREATE_SQL_INSTANCE" = "yes" ] || [ "$CREATE_SQL_INSTANCE" = "y" ]; then
    print_status "Creating Cloud SQL instance '${CLOUD_SQL_INSTANCE}'..." "${GREEN}"
    # Create a small, affordable PostgreSQL instance
    gcloud sql instances create ${CLOUD_SQL_INSTANCE} \
      --database-version=POSTGRES_14 \
      --cpu=1 \
      --memory=3840MB \
      --region=${REGION} \
      --root-password=${DB_PASSWORD:-"changeme"} \
      --storage-size=10GB \
      --storage-type=SSD
      
    # Create the tick_data database
    print_status "Creating database '${DB_NAME}'..." "${GREEN}"
    gcloud sql databases create ${DB_NAME} --instance=${CLOUD_SQL_INSTANCE}
    
    print_status "Cloud SQL instance and database created successfully!" "${GREEN}"
  else
    print_status "Skipping Cloud SQL instance creation." "${YELLOW}"
  fi
fi

# Choose whether to use mock data or real database
print_step "Deployment type selection"
echo "Do you want to use real database connection? (yes/no)"
read USE_REAL_DB

# Default to using mock data
USE_MOCK_DATA="true"

# Option to connect to real database
if [ "$USE_REAL_DB" = "yes" ] || [ "$USE_REAL_DB" = "y" ]; then
  USE_MOCK_DATA="false"
  
  # Ask for database password
  echo -n "Enter your database password (input will be hidden): "
  read -s DB_PASSWORD
  echo ""
  
  # Check if a secret for DB password exists, if not create it
  if ! gcloud secrets describe db-password &>/dev/null; then
    echo "Creating database password secret..."
    echo -n "$DB_PASSWORD" | gcloud secrets create db-password --data-file=-
  else
    echo "Updating database password secret..."
    echo -n "$DB_PASSWORD" | gcloud secrets versions add db-password --data-file=-
  fi
  
  # Get the service account used by Cloud Run
  SERVICE_ACCOUNT=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format="value(spec.template.spec.serviceAccountName)" 2>/dev/null || echo "")
  
  if [ -z "$SERVICE_ACCOUNT" ]; then
    # Use Compute default service account if no specific one is set
    PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    echo "Using default service account: ${SERVICE_ACCOUNT}"
  else
    echo "Using existing service account: ${SERVICE_ACCOUNT}"
  fi
  
  # Grant the service account access to the secret
  echo "Granting the service account access to the secret..."
  gcloud secrets add-iam-policy-binding db-password \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"
  
  echo "Will deploy with real database connection to ${CLOUD_SQL_CONNECTION_NAME}"
  
  # Deploy with database connection
  print_step "Deploying with Cloud SQL connection"
  print_status "Building and deploying Know-Defeat API server with database connection..." "${GREEN}"
  
  # Create a service account for Cloud Run if it doesn't exist
  if ! gcloud iam service-accounts describe ${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    print_status "Creating service account for Cloud Run..." "${YELLOW}"
    gcloud iam service-accounts create ${SERVICE_NAME}-sa \
      --display-name="${SERVICE_NAME} Service Account"
  fi

  # Grant the service account access to Cloud SQL
  print_status "Granting service account Cloud SQL access..." "${YELLOW}"
  gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client" \
    --quiet

  # Grant the service account access to Secret Manager
  print_status "Granting service account Secret Manager access..." "${YELLOW}"
  gcloud secrets add-iam-policy-binding db-password \
    --member="serviceAccount:${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet
  
  # Deploy with the service account and Cloud SQL connection
  print_status "Deploying to Cloud Run..." "${GREEN}"
  gcloud run deploy ${SERVICE_NAME} \
    --source . \
    --platform managed \
    --region ${REGION} \
    --service-account=${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --allow-unauthenticated \
    --add-cloudsql-instances ${CLOUD_SQL_CONNECTION_NAME} \
    --set-env-vars "NODE_ENV=production,USE_MOCK_DATA=${USE_MOCK_DATA},DB_USER=${DB_USER},DB_NAME=${DB_NAME},CLOUD_SQL_CONNECTION_NAME=${CLOUD_SQL_CONNECTION_NAME},DB_HOST=/cloudsql" \
    --update-secrets=DB_PASSWORD=db-password:latest \
    --max-instances=2 \
    --memory=512Mi \
    --timeout=30s
else
  print_step "Deploying with mock data"
  print_status "Will deploy with mock data for testing" "${YELLOW}"
  
  # Create a service account for Cloud Run if it doesn't exist
  if ! gcloud iam service-accounts describe ${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    print_status "Creating service account for Cloud Run..." "${YELLOW}"
    gcloud iam service-accounts create ${SERVICE_NAME}-sa \
      --display-name="${SERVICE_NAME} Service Account"
  fi
  
  # Deploy without database connection
  print_status "Building and deploying Know-Defeat API server with mock data..." "${GREEN}"
  gcloud run deploy ${SERVICE_NAME} \
    --source . \
    --platform managed \
    --region ${REGION} \
    --service-account=${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --allow-unauthenticated \
    --set-env-vars "NODE_ENV=production,USE_MOCK_DATA=${USE_MOCK_DATA}" \
    --max-instances=2 \
    --memory=512Mi \
    --timeout=30s
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)')

print_step "Deployment Summary"
print_status "Deployment complete! Your Know-Defeat API is available at: ${SERVICE_URL}" "${GREEN}"

if [ "$USE_MOCK_DATA" = "false" ]; then
  print_step "Database Schema Initialization"
  print_status "Do you want to initialize the database schema? (yes/no)" "${YELLOW}"
  read INIT_SCHEMA
  
  if [ "$INIT_SCHEMA" = "yes" ] || [ "$INIT_SCHEMA" = "y" ]; then
    print_status "Creating schema file..." "${GREEN}"
    
    # Create temporary schema file with exact schema from docs/database_schema.md
    cat > /tmp/schema.sql << EOF
-- Database Schema for Know-Defeat Trading System

-- tick_data Table
CREATE TABLE IF NOT EXISTS tick_data (
    timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    price DECIMAL(15,6) NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (timestamp, ticker)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tick_data_ticker ON tick_data(ticker);
CREATE INDEX IF NOT EXISTS idx_tick_data_timestamp ON tick_data(timestamp);

-- sim_bots Table
CREATE TABLE IF NOT EXISTS sim_bots (
    bot_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    algorithm_module VARCHAR(255) NOT NULL,
    algorithm_type VARCHAR(50) NOT NULL,
    trade_direction VARCHAR(10) NOT NULL,
    position_size NUMERIC(15,2) NOT NULL,
    trailing_stop_pct NUMERIC(8,6) NOT NULL,
    description TEXT,
    version VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sim_bots_ticker ON sim_bots(ticker);
CREATE INDEX IF NOT EXISTS idx_sim_bots_active ON sim_bots(is_active);

-- sim_bot_trades Table
CREATE TABLE IF NOT EXISTS sim_bot_trades (
    trade_id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL REFERENCES sim_bots(bot_id),
    ticker VARCHAR(10) NOT NULL,
    entry_price NUMERIC(15,6) NOT NULL,
    exit_price NUMERIC(15,6),
    trade_size NUMERIC(15,2) NOT NULL,
    trade_direction VARCHAR(10) NOT NULL,
    entry_time TIMESTAMP NOT NULL DEFAULT NOW(),
    exit_time TIMESTAMP,
    trade_status VARCHAR(20) NOT NULL DEFAULT 'open',
    pnl NUMERIC(15,2),
    pnl_percent NUMERIC(15,6),
    trailing_stop_price NUMERIC(15,6),
    exit_reason VARCHAR(50)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_sim_bot_trades_bot_id ON sim_bot_trades(bot_id);
CREATE INDEX IF NOT EXISTS idx_sim_bot_trades_status ON sim_bot_trades(trade_status);
CREATE INDEX IF NOT EXISTS idx_sim_bot_trades_ticker ON sim_bot_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_sim_bot_trades_entry_time ON sim_bot_trades(entry_time);

-- bot_tick_data Table
CREATE TABLE IF NOT EXISTS bot_tick_data (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    price NUMERIC(15,6) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_bot_id
        FOREIGN KEY(bot_id) 
        REFERENCES sim_bots(bot_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_bot_tick_data_ticker ON bot_tick_data(ticker);
CREATE INDEX IF NOT EXISTS idx_bot_tick_data_bot_id ON bot_tick_data(bot_id);
CREATE INDEX IF NOT EXISTS idx_bot_tick_data_processed ON bot_tick_data(processed);

-- bot_metrics Table
CREATE TABLE IF NOT EXISTS bot_metrics (
    id SERIAL PRIMARY KEY,
    bot_id INTEGER NOT NULL REFERENCES sim_bots(bot_id),
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl NUMERIC(15,2) NOT NULL DEFAULT 0,
    average_pnl_per_trade NUMERIC(15,2) NOT NULL DEFAULT 0,
    win_rate NUMERIC(5,4) NOT NULL DEFAULT 0,
    average_win_amount NUMERIC(15,2) NOT NULL DEFAULT 0,
    average_loss_amount NUMERIC(15,2) NOT NULL DEFAULT 0,
    profit_factor NUMERIC(15,4) NOT NULL DEFAULT 0,
    max_drawdown NUMERIC(15,2) NOT NULL DEFAULT 0,
    sharpe_ratio NUMERIC(10,4) NOT NULL DEFAULT 0,
    risk_reward_ratio NUMERIC(10,4) NOT NULL DEFAULT 0,
    expectancy NUMERIC(10,4) NOT NULL DEFAULT 0,
    rank_score NUMERIC(10,4) NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index for bot_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_bot_metrics_bot_id ON bot_metrics(bot_id);

-- variable_weights Table
CREATE TABLE IF NOT EXISTS variable_weights (
    variable_name VARCHAR(50) PRIMARY KEY,
    weight NUMERIC(5,4) NOT NULL DEFAULT 1.0,
    description TEXT,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Add a few sample bots
INSERT INTO sim_bots (bot_id, name, ticker, algorithm_module, algorithm_type, trade_direction, position_size, trailing_stop_pct, description, version, is_active)
VALUES 
(1, 'TSLA_Breakout_Bot', 'TSLA', 'algorithms.breakout_algorithm', 'breakout', 'BOTH', 1000.0, 0.01, 'TSLA breakout strategy using volatility-based entry', '1.0', true),
(2, 'COIN_Momentum_Bot', 'COIN', 'algorithms.momentum_algorithm', 'momentum', 'LONG', 1000.0, 0.015, 'COIN momentum strategy', '1.0', true),
(3, 'NVDA_Breakout_Bot', 'NVDA', 'algorithms.breakout_algorithm', 'breakout', 'BOTH', 1000.0, 0.01, 'NVDA breakout strategy', '1.0', true),
(4, 'AMD_Momentum_Bot', 'AMD', 'algorithms.momentum_algorithm', 'momentum', 'LONG', 1000.0, 0.012, 'AMD momentum strategy', '1.0', true),
(5, 'AAPL_Support_Resistance_Bot', 'AAPL', 'algorithms.support_resistance_algorithm', 'support_resistance', 'BOTH', 1000.0, 0.008, 'AAPL support resistance strategy', '1.0', true)
ON CONFLICT (bot_id) DO NOTHING;

-- Add sample trades
INSERT INTO sim_bot_trades (trade_id, bot_id, ticker, entry_price, exit_price, trade_size, trade_direction, entry_time, exit_time, trade_status, pnl, pnl_percent, trailing_stop_price, exit_reason)
VALUES 
(1, 1, 'TSLA', 180.25, 185.50, 1000, 'LONG', NOW() - INTERVAL '5 days', NOW() - INTERVAL '4 days', 'closed', 290.83, 0.0291, 183.20, 'trailing_stop'),
(2, 2, 'COIN', 210.75, 206.30, 1000, 'LONG', NOW() - INTERVAL '4 days', NOW() - INTERVAL '3 days', 'closed', -211.39, -0.0211, 206.30, 'trailing_stop'),
(3, 3, 'NVDA', 950.00, 972.25, 1000, 'LONG', NOW() - INTERVAL '3 days', NOW() - INTERVAL '2 days', 'closed', 234.21, 0.0234, 965.00, 'profit_target')
ON CONFLICT (trade_id) DO NOTHING;

-- Add a couple of open trades
INSERT INTO sim_bot_trades (trade_id, bot_id, ticker, entry_price, trade_size, trade_direction, entry_time, trade_status, trailing_stop_price)
VALUES 
(6, 1, 'TSLA', 182.40, 1000, 'LONG', NOW() - INTERVAL '1 day', 'open', 179.75),
(7, 3, 'NVDA', 965.25, 1000, 'LONG', NOW() - INTERVAL '1 day', 'open', 955.60)
ON CONFLICT (trade_id) DO NOTHING;

-- Sample bot metrics
INSERT INTO bot_metrics (bot_id, total_trades, winning_trades, losing_trades, total_pnl, average_pnl_per_trade, win_rate, average_win_amount, average_loss_amount, profit_factor, max_drawdown, sharpe_ratio, risk_reward_ratio, expectancy, rank_score)
VALUES 
(1, 32, 21, 11, 2450.75, 76.59, 0.6563, 152.32, -72.45, 2.10, -450.20, 1.85, 2.10, 0.24, 0.92),
(2, 30, 17, 13, 1650.20, 55.01, 0.5667, 128.75, -75.30, 1.68, -520.10, 1.45, 1.71, 0.19, 0.78),
(3, 28, 18, 10, 2120.50, 75.73, 0.6429, 145.80, -68.90, 1.95, -380.60, 1.72, 2.12, 0.23, 0.89)
ON CONFLICT (bot_id) DO NOTHING;
EOF

    # Apply the schema to the database
    print_status "Applying schema to database..." "${GREEN}"
    gcloud sql import sql ${CLOUD_SQL_INSTANCE} /tmp/schema.sql \
      --database=${DB_NAME} \
      --quiet
    
    # Clean up
    rm /tmp/schema.sql
    
    print_status "Database schema initialization complete!" "${GREEN}"
  else
    print_status "Skipping database schema initialization." "${YELLOW}"
  fi
fi

print_step "Available Endpoints"
echo "  ${SERVICE_URL}/                 - Dashboard page"
echo "  ${SERVICE_URL}/api/bots         - List all trading bots"
echo "  ${SERVICE_URL}/api/trades       - List recent trades"
echo "  ${SERVICE_URL}/api/metrics      - View bot metrics"
echo "  ${SERVICE_URL}/api/dashboard    - Get dashboard summary data"
echo ""
print_status "Try visiting the dashboard in your browser: ${SERVICE_URL}" "${GREEN}"

print_step "Next Steps"
print_status "1. Visit the URL above to verify your deployment" "${YELLOW}"
print_status "2. Read DEPLOYMENT_GUIDE.md for more information on managing your deployment" "${YELLOW}"
print_status "3. If you encounter any issues with Cloud SQL connectivity, run: ./test-db-connection.sh" "${YELLOW}"