#!/usr/bin/env bash
set -euo pipefail

ID_OR_SAFE="${1:?usage: ./assemble.sh <hf_id_or_safe_name>}"
REPO_URL="${REPO_URL:-https://github.com/exchange.git}"
SAFE_NAME="$(echo "$ID_OR_SAFE" | tr '/:' '__')"
WORKDIR="exchange-tmp"
DEST_ROOT="$(pwd)/src/models/$SAFE_NAME"

command -v git >/dev/null
command -v python3 >/dev/null

[ -d "$WORKDIR/.git" ] || git clone "$REPO_URL" "$WORKDIR"
git -C "$WORKDIR" fetch origin || true
git -C "$WORKDIR" checkout main
git -C "$WORKDIR" pull --ff-only || true

read -rs -p "Passphrase: " PASSPHRASE; echo
mkdir -p "$DEST_ROOT"

JOIN="$WORKDIR/tools/join_file.py"
[ -f "$JOIN" ] || { echo "join_file.py missing in repo"; exit 1; }

mapfile -t DIRS < <(find "$WORKDIR/models/$SAFE_NAME" -type d -name "*.chunks" | sort)
for d in "${DIRS[@]}"; do
  MF="$d/manifest.json"
  [ -f "$MF" ] || continue
  OUT="$(python3 - <<'PY' "$MF"
import json,sys,os
with open(sys.argv[1]) as f: m=json.load(f)
print(m.get("original_filename","restored.bin"))
PY
)"
  python3 "$JOIN" "$d" --out "$DEST_ROOT/$OUT" --passphrase "$PASSPHRASE"
done

echo "DONE: assembled into $DEST_ROOT"
