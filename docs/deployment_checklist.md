# Know-Defeat: Google Cloud Run Deployment Checklist

Use this checklist to deploy your Know-Defeat app to Google Cloud Run:

## Prerequisites Setup

- [ ] Google Cloud Platform account with billing enabled
- [ ] Git and GitHub repository configured
- [ ] Git large file issues resolved (see below)

## Fix Git Large File Issues

```bash
# Run our custom cleanup script to remove large files from tracking
./scripts/clean_repo_for_github.sh

# If push still fails, you might need more aggressive cleaning:
git push origin main 2>&1 | grep "error: File" > large_files.txt
cat large_files.txt  # View problematic files
```

## Google Cloud Setup Steps

```bash
# Navigate to your project directory
cd ~/Desktop/CurveAI/Know-Defeat

# Set your project ID and region (use your actual values)
PROJECT_ID=know-defeat-trading
REGION=us-central1  # or your preferred region

# Run the setup script
./scripts/setup_cloud_run.sh
```

## GitHub Repository Configuration

- [ ] Add `GCP_PROJECT_ID` secret (value: "know-defeat-trading")
- [ ] Add `GCP_REGION` secret (value: "us-central1" or your chosen region)
- [ ] Add `GCP_CREDENTIALS` secret (content of credentials.json)
- [ ] Add `DB_PASSWORD` secret (your database password)
- [ ] Add `CLOUD_SQL_CONNECTION_NAME` secret (format: "project:region:instance")

## Deployment Files Check

- [ ] Dockerfile is optimized for Cloud Run
- [ ] GitHub Actions workflow file at `.github/workflows/deploy.yml`
- [ ] App has a health check endpoint at `/healthcheck`
- [ ] Environment variables properly configured in workflow
- [ ] Database connection code supports Cloud SQL

## Final Deployment

```bash
# Add modified files
git add .github/workflows/deploy.yml
git add app/routes/healthcheck.tsx
git add Dockerfile
git add docs/

# Commit changes
git commit -m "Configure Cloud Run deployment with GitHub Actions"

# Push to GitHub to trigger deployment
git push origin main
```

## Post-Deployment Verification

- [ ] Check GitHub Actions tab for deployment progress
- [ ] Verify deployment succeeded
- [ ] Test application at the deployed URL
- [ ] Verify database connection works
- [ ] Check logs in Google Cloud Console for any issues

## Commands Quick Reference

```bash
# View GitHub Actions service account
gcloud iam service-accounts list

# Create database in Cloud SQL
gcloud sql databases create tick_data --instance=know-defeat-db

# View Cloud Run service
gcloud run services list

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=know-defeat"
```