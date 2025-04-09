# Know Defeat Trading System

An advanced algorithmic trading platform with autonomous agent intelligence.

## Project Structure

```
know-defeat/
├── api/                     # Backend API server
├── ui/                      # Frontend Vite app
├── docker/                  # Docker configurations
└── deploy/                  # Deployment scripts
```

## Features

- Real-time market data processing with IB API integration
- Bot management and execution system
- Performance tracking and analysis
- Multiple trading algorithms
- Dashboard visualization
- Fund allocation based on performance

## Development Setup

### Prerequisites

- Node.js v18+
- PostgreSQL 13+
- Docker (optional, for containerized development)

### API Setup

```bash
# Navigate to the API directory
cd api

# Install dependencies
npm install

# Start the development server
npm run dev
```

### UI Setup

```bash
# Navigate to the UI directory
cd ui

# Install dependencies
npm install

# Start the development server
npm run dev
```

### Docker Setup

```bash
# Start both API and UI using Docker Compose
docker-compose up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bots` | GET | List all trading bots |
| `/api/bots/:id` | GET | Get bot details |
| `/api/trades` | GET | List trades (with filtering) |
| `/api/trades/open` | GET | List open trades |
| `/api/metrics` | GET | Get bot performance metrics |
| `/api/dashboard` | GET | Get dashboard summary data |
| `/api/allocation` | GET | Get fund allocation data |

## Deployment

### Google Cloud Setup

```bash
# Create GCP project
gcloud projects create know-defeat-trading --name="Know Defeat Trading"
gcloud config set project know-defeat-trading

# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com
```

### Database Deployment

```bash
# Create Cloud SQL instance
gcloud sql instances create know-defeat-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database and user
gcloud sql databases create tick_data --instance=know-defeat-db
```

### API Deployment

```bash
# Deploy API to Cloud Run
cd api
gcloud run deploy know-defeat-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### UI Deployment

```bash
# Deploy UI to Cloud Run
cd ui
gcloud run deploy know-defeat-ui \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## License

This project is proprietary software owned by Curve AI Solutions.