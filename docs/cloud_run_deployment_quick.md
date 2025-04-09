# Quick Guide: Deploy Know-Defeat to Google Cloud Run

This guide provides a quick, streamlined way to deploy your Remix application to Google Cloud Run using GitHub Actions.

## Step 1: Set Up Google Cloud Project (One-time Setup)

Run this in Google Cloud Shell:

```bash
# Configure variables
export PROJECT_ID=your-project-id
export REGION=your-region  # e.g., us-central1

# Run the setup script
curl -L https://raw.githubusercontent.com/claybowl/Know-Defeat/main/scripts/setup_cloud_run.sh > setup_cloud_run.sh
chmod +x setup_cloud_run.sh
./setup_cloud_run.sh
```

This will:
- Enable required Google Cloud APIs
- Create and configure a service account for GitHub Actions
- Generate credentials for GitHub

## Step 2: Add Secrets to GitHub Repository

In your GitHub repository, go to Settings → Secrets and variables → Actions and add:

- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_CREDENTIALS`: The entire content of the `credentials.json` file
- `GCP_REGION`: Your chosen region (e.g., `us-central1`)
- `DB_PASSWORD`: Your database password (if using Cloud SQL)
- `CLOUD_SQL_CONNECTION_NAME`: Your Cloud SQL instance (if using Cloud SQL)

## Step 3: Push Code to GitHub

The GitHub Actions workflow will automatically:
1. Build your application
2. Package it in a Docker container
3. Deploy it to Cloud Run

You'll find your deployed URL in the GitHub Actions workflow output.

## Handling Large Files

If you have files exceeding GitHub's 100MB limit (like SQL backups):

1. Update your `.gitignore`:
```
*.sql
tick_data_backup.sql
tick_data_backup_clean.sql
```

2. Remove them from Git tracking:
```bash
git rm --cached tick_data_backup.sql
git rm --cached tick_data_backup_clean.sql
```

3. Commit and push:
```bash
git commit -m "Remove large files from Git tracking"
git push origin main
```

For more details, see `docs/github_large_files.md`.

## Common Issues

- **Deployment fails**: Check the GitHub Actions logs for details
- **Database connection issues**: Verify your Cloud SQL connection name and credentials
- **Container crashes**: Check Cloud Run logs in Google Cloud Console

For a more detailed guide, see `docs/gcp-deployment-guide.md`.