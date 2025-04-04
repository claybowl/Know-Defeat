# GCP Deployment Guide for Know Defeat Trading System

This guide outlines the step-by-step process to deploy the Know Defeat algorithmic trading system to Google Cloud Platform (GCP), enabling collaborative development and real-time trading capabilities.

## Table of Contents

1. [GCP Project Setup](#1-gcp-project-setup)
2. [Database Migration](#2-database-migration)
3. [Frontend Deployment](#3-frontend-deployment)
4. [Backend Services](#4-backend-services)
5. [Testing & Verification](#5-testing--verification)
6. [Monitoring Setup](#6-monitoring-setup)
7. [Database Backup & Maintenance](#7-database-backup-and-maintenance)
8. [Security Configuration](#8-security-considerations)
9. [Cost Estimates](#9-estimated-costs)
10. [Partner Access](#10-partner-access-setup)
11. [Implementation Plan](#11-implementation-plan)

## 1. GCP Project Setup

First, create and configure your GCP project:

```bash
# Install Google Cloud SDK if you haven't already
# https://cloud.google.com/sdk/docs/install

# Login to Google Cloud
gcloud auth login

# Create a new GCP project
gcloud projects create know-defeat-trading --name="Know Defeat Trading"

# Set the project as active
gcloud config set project know-defeat-trading

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Setting Up Billing Alert

1. Go to the GCP Console > Billing
2. Create a budget named "Know Defeat Trading Budget" with your $600 credit limit
3. Set alerts at 50%, 75%, and 90% thresholds
4. Configure email notifications for budget alerts

## 2. Database Migration

Set up a Cloud SQL PostgreSQL instance and migrate your data:

```bash
# Create a PostgreSQL instance
gcloud sql instances create trading-db \
  --database-version=POSTGRES_14 \
  --tier=db-g1-small \
  --region=us-central1 \
  --storage-size=10GB \
  --storage-auto-increase \
  --availability-type=zonal \
  --backup-start-time=02:00 \
  --database-flags max_connections=100

# Create a database
gcloud sql databases create tick_data --instance=trading-db

# Set password for the default 'postgres' user
gcloud sql users set-password postgres --instance=trading-db \
  --password=[SECURE_PASSWORD]
```

### Export and Import Data

```bash
# Export schema and data to a SQL file
pg_dump -h localhost -U clayb -d tick_data > tick_data_backup.sql

# Upload the SQL file to a Cloud Storage bucket
gsutil mb gs://know-defeat-trading-backup
gsutil cp tick_data_backup.sql gs://know-defeat-trading-backup/

# Import to Cloud SQL
gcloud sql import sql trading-db \
  gs://know-defeat-trading-backup/tick_data_backup.sql \
  --database=tick_data
```

### Update Database Connection Code

Create a new file `src/cloud_db_connection.py`:

```python
import asyncpg
import os

async def create_db_pool():
    # Get Cloud SQL connection details from environment variables
    db_user = os.environ.get('DB_USER', 'postgres')
    db_password = os.environ.get('DB_PASSWORD', '')
    db_name = os.environ.get('DB_NAME', 'tick_data')
    db_host = os.environ.get('DB_HOST', '127.0.0.1')
    
    # For Cloud SQL Proxy (for development)
    return await asyncpg.create_pool(
        user=db_user,
        password=db_password,
        database=db_name,
        host=db_host,
        port=5432
    )

# For direct connection (for production)
async def create_cloud_db_pool():
    # Format: postgres://{db_user}:{db_pass}@/{db_name}?host=/cloudsql/{connection_name}
    connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')
    db_user = os.environ.get('DB_USER', 'postgres')
    db_password = os.environ.get('DB_PASSWORD', '')
    db_name = os.environ.get('DB_NAME', 'tick_data')
    
    return await asyncpg.create_pool(
        dsn=f"postgres://{db_user}:{db_password}@/{db_name}?host=/cloudsql/{connection_name}"
    )
```

## 3. Frontend Deployment

### Create Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Environment variables for database connection
ENV DB_USER=postgres
ENV DB_NAME=tick_data
ENV DB_HOST=/cloudsql/PROJECT_ID:REGION:trading-db

# Run Streamlit app with proper database connection
CMD ["streamlit", "run", "user_interface/src/streamlit_app2.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Create requirements.txt

```
streamlit>=1.20.0
asyncpg>=0.25.0
pandas>=1.4.0
numpy>=1.21.0
plotly>=5.5.0
psycopg2-binary>=2.9.3
pyyaml>=6.0
scipy>=1.8.0
google-cloud-storage>=2.0.0
```

### Deploy to Cloud Run

```bash
# Create Artifact Registry repository
gcloud artifacts repositories create trading-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repository for Know Defeat Trading"

# Build and push Docker image
gcloud builds submit --tag us-central1-docker.pkg.dev/know-defeat-trading/trading-repo/trading-frontend:v1

# Deploy to Cloud Run with Cloud SQL connection
gcloud run deploy trading-frontend \
  --image us-central1-docker.pkg.dev/know-defeat-trading/trading-repo/trading-frontend:v1 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --add-cloudsql-instances PROJECT_ID:us-central1:trading-db \
  --set-env-vars "DB_USER=postgres,DB_PASSWORD=PASSWORD,DB_NAME=tick_data,CLOUD_SQL_CONNECTION_NAME=PROJECT_ID:us-central1:trading-db"
```

## 4. Backend Services

### Set up Virtual Machine for Trading Bots

```bash
# Create a Compute Engine VM
gcloud compute instances create trading-engine \
  --zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --boot-disk-size=50GB \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --scopes=cloud-platform

# SSH into the VM
gcloud compute ssh trading-engine --zone=us-central1-a
```

### Configure VM Environment

```bash
# Inside the VM, install dependencies
sudo apt-get update && sudo apt-get install -y \
  python3-pip python3-dev build-essential \
  postgresql-client git

# Clone your repository
git clone https://github.com/yourusername/know-defeat-trading.git
cd know-defeat-trading

# Install Python requirements
pip3 install -r requirements.txt

# Install Cloud SQL Proxy
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.0.0/cloud-sql-proxy.linux.amd64
chmod +x cloud-sql-proxy

# Start Cloud SQL Proxy in the background
./cloud-sql-proxy PROJECT_ID:us-central1:trading-db &

# Set up environment variables
export DB_USER=postgres
export DB_PASSWORD=PASSWORD
export DB_NAME=tick_data
export DB_HOST=127.0.0.1
```

### Create Systemd Services

For Trading Bots:

```bash
# Configure systemd service for each bot
sudo bash -c 'cat > /etc/systemd/system/trading-bots.service << EOF
[Unit]
Description=Know Defeat Trading Bots
After=network.target

[Service]
User=USERNAME
WorkingDirectory=/home/USERNAME/know-defeat-trading
ExecStart=/usr/bin/python3 src/run_bots.py
Restart=always
Environment=DB_USER=postgres
Environment=DB_PASSWORD=PASSWORD
Environment=DB_NAME=tick_data
Environment=DB_HOST=127.0.0.1

[Install]
WantedBy=multi-user.target
EOF'

# Enable and start the service
sudo systemctl enable trading-bots
sudo systemctl start trading-bots
```

For Interactive Brokers Controller:

```bash
# Create a systemd service for IB controller
sudo bash -c 'cat > /etc/systemd/system/ib-controller.service << EOF
[Unit]
Description=Interactive Brokers Controller
After=network.target

[Service]
User=USERNAME
WorkingDirectory=/home/USERNAME/know-defeat-trading
ExecStart=/usr/bin/python3 src/ib_controller_simple.py
Restart=always
Environment=DB_USER=postgres
Environment=DB_PASSWORD=PASSWORD
Environment=DB_NAME=tick_data
Environment=DB_HOST=127.0.0.1

[Install]
WantedBy=multi-user.target
EOF'

# Enable and start the service
sudo systemctl enable ib-controller
sudo systemctl start ib-controller
```

## 5. Testing & Verification

Create a test script `test_deployment.py`:

```python
# test_deployment.py
import asyncio
import asyncpg
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database_connection():
    logger.info("Testing database connection...")
    try:
        # Get connection details from environment variables
        db_user = os.environ.get('DB_USER', 'postgres')
        db_password = os.environ.get('DB_PASSWORD', '')
        db_name = os.environ.get('DB_NAME', 'tick_data')
        db_host = os.environ.get('DB_HOST', '127.0.0.1')
        
        # Create connection pool
        pool = await asyncpg.create_pool(
            user=db_user,
            password=db_password,
            database=db_name,
            host=db_host
        )
        
        # Test query
        async with pool.acquire() as conn:
            version = await conn.fetchval("SELECT version()")
            logger.info(f"Connected to PostgreSQL: {version}")
            
            # Test tables
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            logger.info(f"Found {len(tables)} tables:")
            for table in tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['table_name']}")
                logger.info(f"  - {table['table_name']}: {count} rows")
        
        await pool.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

async def test_trading_system():
    logger.info("Testing trading system components...")
    try:
        # Import necessary modules
        sys.path.insert(0, '.')
        from src.bot_ranker import BotRanker
        from src.metrics_calculator import MetricsCalculator
        
        # Get connection pool
        db_user = os.environ.get('DB_USER', 'postgres')
        db_password = os.environ.get('DB_PASSWORD', '')
        db_name = os.environ.get('DB_NAME', 'tick_data')
        db_host = os.environ.get('DB_HOST', '127.0.0.1')
        
        pool = await asyncpg.create_pool(
            user=db_user,
            password=db_password,
            database=db_name,
            host=db_host
        )
        
        # Test BotRanker
        ranker = BotRanker(pool)
        rankings = await ranker.rank_bots()
        logger.info(f"Bot ranker successfully ranked {len(rankings)} bots")
        
        # Test MetricsCalculator
        calc = MetricsCalculator(pool)
        metric_test = await calc.calculate_one_day_performance(1, 1)
        logger.info(f"Metrics calculator test: {metric_test}")
        
        await pool.close()
        return True
    except Exception as e:
        logger.error(f"Trading system test failed: {e}")
        return False

async def main():
    db_success = await test_database_connection()
    if db_success:
        system_success = await test_trading_system()
        if system_success:
            logger.info("All tests passed successfully!")
        else:
            logger.error("Trading system tests failed")
    else:
        logger.error("Database connection tests failed")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:

```bash
# On the VM
python3 test_deployment.py
```

## 6. Monitoring Setup

```bash
# Enable Cloud Monitoring API
gcloud services enable monitoring.googleapis.com

# Create a monitoring dashboard (done through web console)

# Set up a custom uptime check for your frontend
gcloud monitoring uptime-check create http trading-frontend-check \
  --display-name="Trading Frontend Check" \
  --uri=CLOUD_RUN_URL \
  --timeout=10s

# Configure logging for all services
gcloud logging write trading-system-log "Deployment completed successfully" --severity=INFO
```

## 7. Database Backup and Maintenance

Set up scheduled backups:

```bash
# Enable automatic backups (already set during instance creation)

# Set up additional export to Cloud Storage
gcloud scheduler jobs create http tick-data-export \
  --schedule="0 2 * * *" \
  --uri="https://sqladmin.googleapis.com/v1/projects/PROJECT_ID/instances/trading-db/export" \
  --message-body="{\"exportContext\":{\"kind\":\"sql#exportContext\",\"fileType\":\"SQL\",\"uri\":\"gs://know-defeat-trading-backup/backups/tick_data_$(date +%Y%m%d).sql\",\"databases\":[\"tick_data\"]}}" \
  --time-zone="America/New_York" \
  --headers="Content-Type=application/json" \
  --oauth-service-account-email=PROJECT_NUMBER-compute@developer.gserviceaccount.com
```

## 8. Security Considerations

### Service Account Setup

```bash
# Create service account
gcloud iam service-accounts create trading-system-sa \
  --display-name="Trading System Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding know-defeat-trading \
  --member="serviceAccount:trading-system-sa@know-defeat-trading.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding know-defeat-trading \
  --member="serviceAccount:trading-system-sa@know-defeat-trading.iam.gserviceaccount.com" \
  --role="roles/logging.logWriter"

# Create and download a key
gcloud iam service-accounts keys create trading-system-sa-key.json \
  --iam-account=trading-system-sa@know-defeat-trading.iam.gserviceaccount.com
```

### Secure Cloud SQL

```bash
# Disable public IP
gcloud sql instances patch trading-db --no-assign-ip

# Create SSL certificates for secure connections
gcloud sql ssl client-certs create trading-client-cert client-key.pem \
  --instance=trading-db

# Download server certificate
gcloud sql ssl server-certs list --instance=trading-db
```

## 9. Estimated Costs

Monthly cost estimates for this deployment:

| Resource | Specification | Estimated Cost/Month |
|----------|---------------|----------------------|
| Cloud SQL | db-g1-small | $35 |
| Compute Engine | e2-standard-2 | $50 |
| Cloud Run | Pay-per-use | $10-20 |
| Storage and Transfer | Various | $10-20 |
| Cloud Scheduler | Free tier | $0 |
| Logging and Monitoring | Basic usage | $5-10 |
| **Total** | | **$110-135** |

This is well within your $600 credit limit.

## 10. Partner Access Setup

### Add Partner to GCP Project

```bash
gcloud projects add-iam-policy-binding know-defeat-trading \
  --member="user:partner@email.com" \
  --role="roles/editor"
```

### Configure IAP for VM Access

```bash
# Enable IAP API
gcloud services enable iap.googleapis.com

# Configure IAP for the VM
gcloud compute instances add-iam-policy-binding trading-engine \
  --member="user:partner@email.com" \
  --role="roles/iap.tunnelResourceAccessor"
```

### Share Frontend URL

Provide your partner with the Cloud Run frontend URL to access the Streamlit dashboard.

## 11. Implementation Plan

Follow this sequence for a smooth deployment:

1. **Day 1**: Set up GCP project and Cloud SQL
   - Create project and configure billing alerts
   - Deploy Cloud SQL instance
   - Migrate schema and initial data

2. **Day 2**: Deploy frontend and configure network
   - Build and deploy Streamlit UI to Cloud Run
   - Set up networking and secure connections

3. **Day 3**: Set up backend services
   - Deploy VM for trading bots
   - Configure IB Gateway and trading systems
   - Set up systemd services

4. **Day 4**: Testing and monitoring
   - Run comprehensive tests
   - Configure monitoring and alerting
   - Set up database backup schedule

5. **Day 5**: Partner onboarding
   - Configure partner access
   - Document the deployment
   - Conduct knowledge transfer session

## Additional Resources

- [GCP Documentation](https://cloud.google.com/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Compute Engine Documentation](https://cloud.google.com/compute/docs)

## 12. Remix.js Frontend Deployment

For the Remix.js frontend (instead of Streamlit), additional steps are needed:

### Create a Node.js Dockerfile

Create a `Dockerfile` for the Remix app in the project root:

```dockerfile
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# Production image, copy all the files and run
FROM base AS runner
WORKDIR /app
ENV NODE_ENV production

# Copy built assets from builder stage
COPY --from=builder /app/build ./build
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules

# Install only production dependencies
RUN npm prune --production

# Expose port
EXPOSE 8080

# Set environment variables for database connection
ENV PORT=8080
ENV DB_USER=postgres
ENV DB_NAME=tick_data
ENV DB_HOST=/cloudsql/PROJECT_ID:REGION:trading-db

# Run the app
CMD ["node", "./build/server.js"]
```

### Configure Cloud Build for Remix

Create a `cloudbuild.yaml` file:

```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'us-central1-docker.pkg.dev/$PROJECT_ID/trading-repo/trading-remix-frontend:$COMMIT_SHA', '.']
  
  # Push the container image to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'us-central1-docker.pkg.dev/$PROJECT_ID/trading-repo/trading-remix-frontend:$COMMIT_SHA']
  
  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'trading-remix-frontend'
      - '--image=us-central1-docker.pkg.dev/$PROJECT_ID/trading-repo/trading-remix-frontend:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--add-cloudsql-instances=$PROJECT_ID:us-central1:trading-db'
      - '--set-env-vars=DB_USER=postgres,DB_PASSWORD=$$DB_PASSWORD,DB_NAME=tick_data,CLOUD_SQL_CONNECTION_NAME=$PROJECT_ID:us-central1:trading-db'
      - '--min-instances=1'
      - '--max-instances=5'

substitutions:
  _PROJECT_ID: 'know-defeat-trading'

options:
  logging: CLOUD_LOGGING_ONLY

secretEnv: ['DB_PASSWORD']

availableSecrets:
  secretManager:
  - versionName: projects/$PROJECT_ID/secrets/db-password/versions/latest
    env: 'DB_PASSWORD'
```

### Set up API Endpoints for Remix

Create API endpoints in your Remix app to interface with your PostgreSQL database:

```javascript
// app/routes/api/bots.js
import { json } from "@remix-run/node";
import { db } from "~/lib/db.server";

export async function loader({ request }) {
  const bots = await db.query(
    "SELECT bot_id, name, ticker, algorithm_type FROM sim_bots WHERE is_active = true"
  );
  
  return json(bots);
}
```

### Configure Cloud SQL Connection from Node.js

Update `app/lib/db.server.js`:

```javascript
import { Pool } from 'pg';

let pool;

// Initialize the connection pool
if (!pool) {
  const isProduction = process.env.NODE_ENV === 'production';
  
  if (isProduction && process.env.CLOUD_SQL_CONNECTION_NAME) {
    // Production environment with Cloud SQL
    pool = new Pool({
      user: process.env.DB_USER,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      host: process.env.DB_HOST.startsWith('/cloudsql') 
        ? process.env.DB_HOST 
        : '/cloudsql/' + process.env.CLOUD_SQL_CONNECTION_NAME,
    });
  } else {
    // Development environment
    pool = new Pool({
      user: process.env.DB_USER || 'clayb',
      password: process.env.DB_PASSWORD || 'musicman',
      database: process.env.DB_NAME || 'tick_data',
      host: process.env.DB_HOST || 'localhost',
      port: 5432,
    });
  }
}

export const db = {
  query: (text, params) => pool.query(text, params),
  getClient: () => pool.connect(),
};
```

## 13. Continuous Integration/Deployment

Set up automated deployments with GitHub:

```bash
# Create a Cloud Build trigger linked to your GitHub repository
gcloud builds triggers create github \
  --name="trading-frontend-deploy" \
  --repo="yourusername/know-defeat-trading" \
  --branch-pattern="main" \
  --build-config="cloudbuild.yaml"
```

## 14. Disaster Recovery Plan

### Database Disaster Recovery

```bash
# Create a cross-region backup policy
gcloud sql instances patch trading-db \
  --backup-location=us-west1 \
  --backup-retention-settings-retained-backups=7

# Configure point-in-time recovery
gcloud sql instances patch trading-db \
  --enable-point-in-time-recovery

# Test restore procedure (document for partners)
echo "In case of database disaster:

1. Access Cloud SQL in GCP Console
2. Select 'trading-db' instance
3. Go to 'Backups' tab
4. Select the most recent backup
5. Click 'Restore'
6. Specify a new instance name
7. Wait for restore to complete
8. Update connection strings in Cloud Run"
```

### Application Disaster Recovery

```bash
# Create a secondary region for Cloud Run deployment
gcloud run deploy trading-frontend-dr \
  --image us-central1-docker.pkg.dev/know-defeat-trading/trading-repo/trading-frontend:v1 \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --add-cloudsql-instances PROJECT_ID:us-central1:trading-db \
  --set-env-vars "DB_USER=postgres,DB_PASSWORD=PASSWORD,DB_NAME=tick_data,CLOUD_SQL_CONNECTION_NAME=PROJECT_ID:us-central1:trading-db"
```
