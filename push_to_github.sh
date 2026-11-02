#!/usr/bin/env bash
set -e
REPO_URL="${1:-https://github.com/k95122256-source/macro-tracker.git}"
echo "🚀 Deploying Macro Tracker v2.0 (11,785-Day Simulation + 365-Day Rolling DB) to $REPO_URL ..."

if [ ! -d ".git" ]; then
  git init
  git branch -M main
  git remote add origin "$REPO_URL" || git remote set-url origin "$REPO_URL"
fi

git add LICENSE README.md index.html update_macro_data.py requirements.txt macro_data.json latest_macro.json archive/ data/ .github/workflows/daily_macro_sync.yml
git commit -m "feat: upgrade to v2.0 engine with 11,785-day simulation and 365-day rolling DB" || true
git push -u origin main --force
echo "✅ GitHub 업로드가 완료되었습니다! (Repository: $REPO_URL)"
