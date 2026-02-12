#!/usr/bin/env bash
set -euo pipefail

# Push current migration repo to specified GitHub URL.
# Usage:
#   bash scripts/push_to_backup_repo.sh

REMOTE_URL="https://github.com/Ethan-Li-k/BackupDribbling.git"
BRANCH="master"

git -C /root/dribbling_migration_repo rev-parse --is-inside-work-tree >/dev/null

if git -C /root/dribbling_migration_repo remote get-url origin >/dev/null 2>&1; then
  git -C /root/dribbling_migration_repo remote set-url origin "${REMOTE_URL}"
else
  git -C /root/dribbling_migration_repo remote add origin "${REMOTE_URL}"
fi

git -C /root/dribbling_migration_repo push -u origin "${BRANCH}"
