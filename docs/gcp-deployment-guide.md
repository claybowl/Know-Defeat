# Know-Defeat: Google Cloud Run Deployment Guide

This guide provides step-by-step instructions for deploying the Know-Defeat trading application to Google Cloud Run.

## Prerequisites

- Google Cloud Platform account
- GitHub repository with your Know-Defeat codebase
- Git installed on your local machine
- Google Cloud CLI (optional for local setup)

## Initial Setup (One-time Configuration)

### Step 1: Set Up Google Cloud Project

1. **Create or select a Google Cloud project**:
   - Log in to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Note your `PROJECT_ID` for later use

2. **Enable required APIs**:
   - Open Cloud Shell and run these commands (replacing `<your-project-id>` with your actual project ID):

```bash
export PROJECT_ID=<your-project-id>
export REGION=<your-region>  # e.g., us-central1, europe-west1
export APP_NAME=know-defeat

# Enable required Google Cloud APIs
gcloud services enable artifactregistry.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable sqladmin.googleapis.com  # For Cloud SQL if needed
```

### Step 2: Create Service Account for GitHub Actions

```bash
# Create a service account
gcloud iam service-accounts create github-actions

# Assign necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# For Cloud SQL access (if using Cloud SQL)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Create a key file for GitHub Actions
gcloud iam service-accounts keys create credentials.json \
  --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

3. **View and copy the credentials**:
```bash
cat credentials.json
```
   - Copy the entire contents of this file for the next step

### Step 3: Configure GitHub Repository Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Create the following secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_CREDENTIALS`: The entire content of the `credentials.json` file
   - `GCP_REGION`: Your chosen region (e.g., `us-central1`)

### Step 4: Set Up Cloud SQL (If Needed)

If you're using a PostgreSQL database:

1. **Create a Cloud SQL instance**:
```bash
gcloud sql instances create know-defeat-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=$REGION \
  --root-password=<secure-password>
```

2. **Create a database**:
```bash
gcloud sql databases create tick_data --instance=know-defeat-db
```

3. **Configure user**:
```bash
gcloud sql users create postgres \
  --instance=know-defeat-db \
  --password=<secure-user-password>
```

4. **Note the connection details**:
   - Instance connection name: `PROJECT_ID:REGION:know-defeat-db`
   - Save this for the GitHub Actions workflow

## Deployment Configuration

### Step 1: Create GitHub Actions Workflow File

Create a file at `.github/workflows/deploy.yml` in your repository:

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches:
      - main

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  GAR_LOCATION: ${{ secrets.GCP_REGION }}
  SERVICE: know-defeat
  REGION: ${{ secrets.GCP_REGION }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
        with:
          install_components: 'beta'

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: '${{ secrets.GCP_CREDENTIALS }}'

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Authorize Docker push
        run: gcloud auth configure-docker ${{ env.GAR_LOCATION }}-docker.pkg.dev

      - name: Build and push container
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ env.GAR_LOCATION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/cloud-run-source-deploy/${{ env.SERVICE }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Deploy to Cloud Run
        id: deploy
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: ${{ env.SERVICE }}
          region: ${{ env.REGION }}
          image: ${{ env.GAR_LOCATION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/cloud-run-source-deploy/${{ env.SERVICE }}:${{ github.sha }}
          flags: |
            --port=8080
            --allow-unauthenticated
            --set-env-vars=NODE_ENV=production
            --set-env-vars=DB_USER=postgres
            --set-env-vars=DB_NAME=tick_data
            --set-env-vars=DB_PASSWORD=${{ secrets.DB_PASSWORD }}
            --set-env-vars=CLOUD_SQL_CONNECTION_NAME=${{ secrets.CLOUD_SQL_CONNECTION_NAME }}

      - name: Show output
        run: echo ${{ steps.deploy.outputs.url }}
```

### Step 2: Setup Cloud SQL Connection (If needed)

To connect your Remix app to Cloud SQL:

1. Add the following secrets to GitHub:
   - `DB_PASSWORD`: Your database password
   - `CLOUD_SQL_CONNECTION_NAME`: Your Cloud SQL instance connection name (format: `PROJECT_ID:REGION:know-defeat-db`)

2. Make sure your app is configured to use Cloud SQL in production mode. The Dockerfile should correctly build and serve your Remix application.

## Deployment Process

1. **Commit and push your changes**:
```bash
git add .
git commit -m "Add Cloud Run deployment configuration"
git push origin main
```

2. **Monitor the deployment**:
   - Go to GitHub repository → Actions tab
   - You'll see the workflow running
   - Check for any errors in the logs

3. **Access your deployed application**:
   - Once deployment completes, find the URL in the GitHub Actions output
   - The URL will be in the format: `https://know-defeat-HASH-REGION.a.run.app`

## Troubleshooting

### Connection Issues to Cloud SQL

If your app can't connect to Cloud SQL:

1. Verify the `CLOUD_SQL_CONNECTION_NAME` is correct
2. Check that Cloud SQL Admin API is enabled
3. Ensure the Cloud Run service account has Cloud SQL Client role

### Container Errors

If your container fails to start:

1. Check Cloud Run logs in Google Cloud Console
2. Verify environment variables are correctly set
3. Test the container locally before deployment:
```bash
docker build -t know-defeat .
docker run -p 8080:8080 know-defeat
```

### Database Migration Issues

For database initialization or migration:

1. Consider creating a separate workflow for database schema setup
2. Use scripts to initialize your database structure
3. Backup your data before any major migrations

## Keeping Costs Low

To minimize GCP costs:

1. Use the smallest Cloud Run instance configuration that meets your needs
2. Choose a db-f1-micro tier for Cloud SQL during development
3. Set minimum instances to 0 to allow scale-to-zero when not in use
4. Monitor your billing dashboard regularly

## Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL Documentation](https://cloud.google.com/sql/docs)
- [GitHub Actions for Google Cloud](https://github.com/google-github-actions)
- [Remix Deployment Documentation](https://remix.run/docs/en/main/guides/deployment)