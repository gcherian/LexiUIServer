#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: ./compose.sh <hf_model_id>}"
REPO_URL="${REPO_URL:-https://github.com/exchange.git}"
SAFE_NAME="$(echo "$MODEL_ID" | tr '/:' '__')"
WORKDIR="exchange-tmp"
VENV=".ml-venv"
UL_BRANCH="upload_$(date +%s)"
BATCH_N="${BATCH_N:-1}"   # parts per commit (1=smallest pushes)

read -rs -p "Passphrase: " PASSPHRASE; echo

command -v git >/dev/null
command -v python3 >/dev/null

[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python3 -m pip -q install --upgrade pip
python3 -m pip -q install "huggingface_hub>=0.23" "cryptography>=42.0.0"

[ -d "$WORKDIR/.git" ] || git clone "$REPO_URL" "$WORKDIR"

git -C "$WORKDIR" config http.sslBackend secure-transport || true
git -C "$WORKDIR" config http.version HTTP/1.1
git -C "$WORKDIR" config pack.window 0
git -C "$WORKDIR" config pack.depth 0
git -C "$WORKDIR" config pack.threads 1
git -C "$WORKDIR" config core.compression 9
git -C "$WORKDIR" config http.expect false
git -C "$WORKDIR" config http.lowSpeedLimit 0
git -C "$WORKDIR" config http.lowSpeedTime 999999

mkdir -p "$WORKDIR/models/$SAFE_NAME" "$WORKDIR/tools"

DL_TMP="$(python3 - <<'PY'
import tempfile; print(tempfile.mkdtemp(prefix="hf-snap-"))
PY
)"

python3 - "$MODEL_ID" "$DL_TMP" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  ignore_patterns=["*.md",".gitattributes",".git/*","**/.git/*"])
PY

# --- chunking/encryption ---
python3 - "$DL_TMP" "$WORKDIR/models/$SAFE_NAME" "$PASSPHRASE" <<'PY'
# (same Python chunking code as before)
PY

# --- write join_file.py into repo ---
python3 - "$WORKDIR/tools/join_file.py" <<'PY'
# (same join_file.py code as before)
PY
chmod +x "$WORKDIR/tools/join_file.py"

printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n" ".ml-venv/" "__pycache__/" "*.cache/" "*.tmp" ".DS_Store" "hf-snap-*/" ".parts.*" > "$WORKDIR/.gitignore"

cd "$WORKDIR"
git fetch origin || true

git checkout --orphan "$UL_BRANCH"
git rm -r --cached . 2>/dev/null || true
git add .gitignore tools/join_file.py
git commit -m "init: ignore + join tool"
git push -u origin "$UL_BRANCH"

# --- push parts in batches ---
CNT=0
for f in $(find "models/$SAFE_NAME" -type f -name "*.part*" | sort); do
  git add "$f"
  CNT=$((CNT+1))
  if [ "$CNT" -ge "$BATCH_N" ]; then
    git commit -m "$SAFE_NAME: add parts"
    git push origin "$UL_BRANCH"
    CNT=0
  fi
done
if [ "$CNT" -gt 0 ]; then
  git commit -m "$SAFE_NAME: add parts (tail)"
  git push origin "$UL_BRANCH"
fi

# --- push manifests & metadata ---
SMALL_FILES=$(find "models/$SAFE_NAME" -type f \( -name "manifest.json" -o -name "*.json" ! -name "*.part*" \) | sort)
if [ -n "$SMALL_FILES" ]; then
  git add $SMALL_FILES
  git commit -m "$SAFE_NAME: add manifests & metadata" || true
  git push origin "$UL_BRANCH"
fi

# --- replace main with this upload branch ---
git push origin "$UL_BRANCH:main" --force
git push origin --delete "$UL_BRANCH" || true

if [ "${CLEAN_REMOTE:-0}" = "1" ]; then
  for b in $(git ls-remote --heads origin | awk '{print $2}' | sed 's#refs/heads/##'); do
    if [ "$b" != "main" ]; then git push origin --delete "$b" || true; fi
  done
fi

echo "DONE: main overwritten with $SAFE_NAME on $REPO_URL"
