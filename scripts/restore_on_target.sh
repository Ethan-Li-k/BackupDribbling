#!/usr/bin/env bash
set -euo pipefail

# One-click restore for target machine (IsaacLab environment only)
# Usage:
#   bash scripts/restore_on_target.sh \
#     --bundle-dir /path/to/dribbling_migration_repo \
#     --workspace-root /root \
#     --run-id 2025-12-11_18-01-51 \
#     --checkpoint model_1300.pt \
#     --use-runtime-exceptions true

BUNDLE_DIR=""
WORKSPACE_ROOT="/root"
RUN_ID="2025-12-11_18-01-51"
CHECKPOINT="model_1300.pt"
USE_RUNTIME_EXCEPTIONS="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --workspace-root) WORKSPACE_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --use-runtime-exceptions) USE_RUNTIME_EXCEPTIONS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$BUNDLE_DIR" ]]; then
  echo "ERROR: --bundle-dir is required"
  exit 1
fi

SOCERLAB_ROOT="${WORKSPACE_ROOT}/soccerLab"
TARGET_SRC="${SOCERLAB_ROOT}/source/soccerTask"
TARGET_TRAIN="${SOCERLAB_ROOT}/scripts/rsl_rl/base/train.py"
TARGET_LOG="${WORKSPACE_ROOT}/logs/rsl_rl/dribbling_g1/${RUN_ID}"
TARGET_T1="${WORKSPACE_ROOT}/data/assets/assetslib/T1"

mkdir -p "${SOCERLAB_ROOT}/source"
mkdir -p "$(dirname "${TARGET_TRAIN}")"
mkdir -p "$(dirname "${TARGET_LOG}")"
mkdir -p "${TARGET_T1}"

# 1) copy strict files
if [[ ! -d "${BUNDLE_DIR}/files_strict/source/soccerTask" ]]; then
  echo "ERROR: strict source bundle not found"
  exit 1
fi
rsync -a "${BUNDLE_DIR}/files_strict/source/soccerTask/" "${TARGET_SRC}/"

# 2) optional runtime exceptions (recommended for runnable dribbling)
if [[ "${USE_RUNTIME_EXCEPTIONS}" == "true" ]]; then
  if [[ -d "${BUNDLE_DIR}/files_runtime_exceptions/source/soccerTask" ]]; then
    rsync -a "${BUNDLE_DIR}/files_runtime_exceptions/source/soccerTask/" "${TARGET_SRC}/"
  fi
  if [[ -f "${BUNDLE_DIR}/files_runtime_exceptions/source/soccerLab/scripts/rsl_rl/base/train.py" ]]; then
    rsync -a "${BUNDLE_DIR}/files_runtime_exceptions/source/soccerLab/scripts/rsl_rl/base/train.py" "${TARGET_TRAIN}"
  fi
  if [[ -f "${BUNDLE_DIR}/files_runtime_exceptions/data/assets/assetslib/T1/T1.urdf" ]]; then
    rsync -a "${BUNDLE_DIR}/files_runtime_exceptions/data/assets/assetslib/T1/T1.urdf" "${TARGET_T1}/T1.urdf"
  fi
fi

# 3) restore logs/checkpoint snapshot
mkdir -p "${TARGET_LOG}/params"
rsync -a "${BUNDLE_DIR}/files_strict/logs/rsl_rl/dribbling_g1/${RUN_ID}/" "${TARGET_LOG}/"

# 4) install task package
python -m pip install -e "${TARGET_SRC}"

# 5) preflight checks
python - <<PY
import os, sys
req = [
    "${TARGET_SRC}/soccerTask/__init__.py",
    "${TARGET_SRC}/soccerTask/train/dribbling/mdp/rewards.py",
    "${TARGET_LOG}/params/env.yaml",
    "${TARGET_LOG}/${CHECKPOINT}",
]
missing = [p for p in req if not os.path.exists(p)]
if missing:
    print("Missing files:")
    for p in missing:
        print(" -", p)
    sys.exit(2)
print("Preflight OK")
PY

cat <<EOF

Restore done.

Run (no video):
python ${WORKSPACE_ROOT}/soccerLab/scripts/rsl_rl/base/train.py \
  --task Loco-G1-Dribbling \
  --headless \
  --resume True \
  --load_run ${RUN_ID} \
  --checkpoint ${CHECKPOINT}

EOF
