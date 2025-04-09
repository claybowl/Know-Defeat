# Know-Defeat Trading System: Development & Deployment Plan

## 1. Project Structure

```
know-defeat/
├── api/                     # Backend API server
├── ui/                      # Frontend Vite app
├── docker/                  # Docker configurations
└── deploy/                  # Deployment scripts
```

## 2. Development Roadmap

### Phase 1: Foundation (Week 1)
- **Day 1-2: Initial Setup**
  - Create Vite frontend project with TypeScript
  - Set up Express API server
  - Configure Docker development environment
  - Set up database connection module

- **Day 3-5: Core Infrastructure**
  - Build API endpoints for bots, trades, metrics
  - Create authentication middleware
  - Implement Mock API mode for frontend development
  - Set up frontend routing and basic layouts

### Phase 2: Core Features (Week 2)
- **Day 1-3: Dashboard & Bot Management**
  - Create dashboard with system statistics
  - Build bot listing and detail views
  - Implement data fetching with React Query
  - Develop bot metrics visualization

- **Day 4-5: Trade Monitoring**
  - Build active trades view
  - Create trade history view with filtering
  - Implement trade analytics charts

### Phase 3: Advanced Features (Week 3)
- **Day 1-3: Fund Allocation System**
  - Create allocation visualization
  - Build bot ranking system interface
  - Implement allocation management UI

- **Day 4-5: Testing & Optimization**
  - End-to-end testing of all features
  - Performance optimization
  - Responsive design for all screen sizes
  - Improve loading states and error handling

### Phase 4: Deployment (Week 4)
- **Day 1-2: Cloud Infrastructure**
  - Configure Google Cloud Project
  - Set up Cloud SQL database instance
  - Create Cloud Run service configurations

- **Day 3-5: Deployment & Documentation**
  - Deploy database, API, and UI components
  - Set up CI/CD pipeline
  - Create deployment documentation
  - Knowledge transfer and handover

## 3. Technical Implementation

### Frontend (Vite + React)
```bash
# Create Vite project
npm create vite@latest know-defeat-ui -- --template react-ts
cd know-defeat-ui

# Install dependencies
npm install react-router-dom @mui/material @emotion/react @emotion/styled
npm install recharts axios @tanstack/react-query
```

### Backend (Express)
```bash
# Create Express project
mkdir know-defeat-api
cd know-defeat-api
npm init -y

# Install dependencies
npm install express cors pg dotenv helmet
npm install typescript @types/express @types/node -D

# Create TypeScript config
npx tsc --init
```

### Docker Configuration
```dockerfile
# api/Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 8080
CMD ["node", "dist/server.js"]
```

```dockerfile
# ui/Dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 4. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bots` | GET | List all trading bots |
| `/api/bots/:id` | GET | Get bot details |
| `/api/trades` | GET | List trades (with filtering) |
| `/api/trades/open` | GET | List open trades |
| `/api/metrics` | GET | Get bot performance metrics |
| `/api/dashboard` | GET | Get dashboard summary data |
| `/api/allocation` | GET | Get fund allocation data |

## 5. Deployment Strategy

### 1. Google Cloud Setup
```bash
# Create GCP project (if not already created)
gcloud projects create know-defeat-trading --name="Know Defeat Trading"
gcloud config set project know-defeat-trading

# Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com sqladmin.googleapis.com
```

### 2. Database Deployment
```bash
# Create Cloud SQL instance
gcloud sql instances create know-defeat-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --root-password=secure-password

# Create database and user
gcloud sql databases create tick_data --instance=know-defeat-db
```

### 3. API Deployment
```bash
# Deploy API to Cloud Run
gcloud run deploy know-defeat-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "DB_HOST=/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME,DB_USER=xxx,DB_NAME=tick_data"
```

### 4. Frontend Deployment
```bash
# Deploy frontend to Cloud Run
gcloud run deploy know-defeat-ui \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "VITE_API_URL=https://know-defeat-api-xxxx.run.app"
```

## 6. Cloud Run CI/CD Automation

```yaml
# cloudbuild.yaml
steps:
  # Build and deploy API
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/know-defeat-api', './api']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: ['run', 'deploy', 'know-defeat-api', '--image', 'gcr.io/$PROJECT_ID/know-defeat-api', '--region', 'us-central1', '--platform', 'managed']

  # Build and deploy UI
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/know-defeat-ui', './ui']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args: ['run', 'deploy', 'know-defeat-ui', '--image', 'gcr.io/$PROJECT_ID/know-defeat-ui', '--region', 'us-central1', '--platform', 'managed']
```

## 7. Local Development Setup

```yaml
# docker-compose.yml
version: '3'

services:
  api:
    build: ./api
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=host.docker.internal
      - DB_PORT=5432
      - DB_USER=clayb
      - DB_PASSWORD=musicman
      - DB_NAME=tick_data
    volumes:
      - ./api:/app
      - /app/node_modules

  ui:
    build: ./ui
    ports:
      - "3000:80"
    volumes:
      - ./ui:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8080
```

This plan provides a complete roadmap for building and deploying your decoupled trading system with a clear separation between the frontend, API, and database layers.