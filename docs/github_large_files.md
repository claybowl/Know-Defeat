# Handling Large Files in Git/GitHub

This guide addresses how to handle large files (like SQL backups) in your repository to avoid GitHub size limits.

## The Problem

GitHub has a file size limit of 100MB. Large files in your project (like `tick_data_backup.sql` and `tick_data_backup_clean.sql`) exceed this limit and prevent pushing to GitHub.

## Solution 1: Git LFS (Large File Storage)

Git LFS is designed for large files:

1. **Install Git LFS**:
   ```bash
   # Install Git LFS
   git lfs install
   ```

2. **Track large files**:
   ```bash
   # Track SQL files
   git lfs track "*.sql"
   
   # Add .gitattributes to repository
   git add .gitattributes
   git commit -m "Configure Git LFS for large SQL files"
   ```

3. **Move existing files to LFS**:
   If files are already in your repository:
   ```bash
   git lfs migrate import --include="*.sql" --everything
   ```

4. **Push changes**:
   ```bash
   git push origin main
   ```

## Solution 2: Exclude Large Files

If you don't need the large files in your repository:

1. **Update .gitignore**:
   ```bash
   # Add to .gitignore
   echo "*.sql" >> .gitignore
   echo "tick_data_backup.sql" >> .gitignore
   echo "tick_data_backup_clean.sql" >> .gitignore
   ```

2. **Remove files from Git tracking**:
   ```bash
   git rm --cached tick_data_backup.sql
   git rm --cached tick_data_backup_clean.sql
   git commit -m "Remove large SQL files from Git tracking"
   ```

3. **Push changes**:
   ```bash
   git push origin main
   ```

## Solution 3: Use Database Dumps Alternative

Instead of storing large SQL files directly:

1. **Create smaller samples**:
   - Create a smaller, representative sample of your database for testing
   - Use scripts to generate schema-only SQL files (no data)

2. **Use database migration tools**:
   - Store migrations/schema changes instead of full dumps
   - Consider tools like Flyway, Liquibase, or Prisma Migrate

3. **Store full backups elsewhere**:
   - Use Google Cloud Storage or similar service for large backups
   - Create scripts to automate backup/restore

## For Current Repository

If you're having trouble with an existing repository:

1. **Create a clean clone**:
   ```bash
   # Clone without large history
   git clone --depth 1 https://github.com/yourusername/Know-Defeat.git know-defeat-clean
   
   # Remove .git directory
   cd know-defeat-clean
   rm -rf .git
   
   # Initialize new repository
   git init
   git add .
   git commit -m "Initial commit"
   
   # Add GitHub remote
   git remote add origin https://github.com/yourusername/Know-Defeat.git
   ```

2. **Force push to GitHub** (use with caution as it rewrites history):
   ```bash
   git push -f origin main
   ```

## Best Practices

1. **Database setup scripts**:
   - Include schema-only SQL files (no data)
   - Add seed data scripts separately, keeping them small

2. **Documentation**:
   - Document where full backups are stored
   - Include instructions for setting up databases from scratch

3. **Automated testing**:
   - Use GitHub Actions to create test databases
   - Set up scripts to validate schemas during CI/CD