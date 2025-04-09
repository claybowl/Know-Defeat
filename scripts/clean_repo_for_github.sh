#!/bin/bash
# Clean repository for GitHub deployment by removing large files

# Exit on error
set -e

echo "=== Know-Defeat Repository Cleaning Tool ==="
echo "This script will help remove large files from Git tracking"
echo ""

# Make sure we're in the repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
echo "Working in repository: $REPO_ROOT"
echo ""

# Remove large files from Git tracking
echo "Removing large files from Git tracking..."

# Log files
echo "Removing log files..."
git rm --cached ib_controller_simple.log 2>/dev/null || echo "- ib_controller_simple.log not tracked"
git rm --cached trading_system.log 2>/dev/null || echo "- trading_system.log not tracked"
git rm --cached src/ib_controller_simple.log 2>/dev/null || echo "- src/ib_controller_simple.log not tracked"
git rm --cached logs/app_logs/ib_controller.log 2>/dev/null || echo "- logs/app_logs/ib_controller.log not tracked"

# CSV data files
echo "Removing large data files..."
git rm --cached data/tick_data.csv 2>/dev/null || echo "- data/tick_data.csv not tracked"

# SQL files
echo "Removing SQL backup files..."
git rm --cached tick_data_backup.sql 2>/dev/null || echo "- tick_data_backup.sql not tracked"
git rm --cached tick_data_backup_clean.sql 2>/dev/null || echo "- tick_data_backup_clean.sql not tracked"

# Update .gitignore
echo "Updating .gitignore file..."
cat > .gitignore << EOL
# System and IDE files
.DS_Store
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log
ib_controller*.log
trading_system.log

# Large data files
data/*.csv
data/*.dump
*.dump
*.backup
*.csv

# SQL files
*.sql
tick_data_backup.sql
tick_data_backup_clean.sql

# Node.js
node_modules/
npm-debug.log
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.coverage

# Build files
/build/
/dist/
/.cache/

# Cloud-specific
.gcloudignore
credentials.json

# Remix
/public/build

# Cursor
.cursor/
EOL

echo "Committing changes..."
git add .gitignore
git commit -m "Update .gitignore to exclude large files" || echo "No changes to commit"

echo ""
echo "=== REPOSITORY CLEANING COMPLETE ==="
echo ""
echo "Next steps:"
echo "1. Push your changes to GitHub:"
echo "   git push origin main"
echo ""
echo "2. If push fails due to large files in history, you may need to create a fresh repository:"
echo "   - Create a new empty repository on GitHub"
echo "   - Push only the current state without history:"
echo "     git push --force origin main"
echo ""
echo "3. If you still have issues, consider cloning without history:"
echo "   git clone --depth 1 https://github.com/yourusername/Know-Defeat.git know-defeat-clean"
echo "   cd know-defeat-clean"
echo "   rm -rf .git"
echo "   git init"
echo "   git add ."
echo "   git commit -m \"Initial commit\""
echo "   git remote add origin https://github.com/yourusername/Know-Defeat.git"
echo "   git push -f origin main"
echo ""