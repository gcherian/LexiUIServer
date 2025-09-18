#!/usr/bin/env bash
set -euo pipefail

ID_OR_SAFE="${1:?usage: ./assemble.sh <hf_id_or_safe_name>}"
REPO_URL="${REPO_URL:-https://github.com/exchange.git}"

if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "Python not found"; exit 1
fi

SAFE_NAME="$(echo "$ID_OR_SAFE" | tr '/:' '__')"
WORKDIR="exchange-tmp"
SRC_ROOT="$WORKDIR/models/$SAFE_NAME"
DEST_ROOT="$(pwd)/src/models/$SAFE_NAME"

command -v git >/dev/null

[ -d "$WORKDIR/.git" ] || git clone "$REPO_URL" "$WORKDIR"
git -C "$WORKDIR" config core.autocrlf false
git -C "$WORKDIR" config core.eol lf
git -C "$WORKDIR" fetch origin || true
git -C "$WORKDIR" checkout main
git -C "$WORKDIR" pull --ff-only || true

JOIN="$WORKDIR/tools/join_file.py"
[ -f "$JOIN" ] || { echo "join_file.py missing at $JOIN"; exit 1; }

read -rs -p "Passphrase: " PASSPHRASE; echo
mkdir -p "$DEST_ROOT"

# copy all non-chunk files
"$PY" - <<'PY' "$SRC_ROOT" "$DEST_ROOT"
import os, shutil, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
for p in src.rglob("*"):
    rel = p.relative_to(src)
    if any(str(part).endswith(".chunks") for part in rel.parts):
        continue
    if p.is_dir():
        (dst/rel).mkdir(parents=True, exist_ok=True)
    elif p.is_file():
        (dst/rel.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst/rel)
PY

# join each *.chunks
find "$SRC_ROOT" -type d -name "*.chunks" | sort | while IFS= read -r d; do
  MF="$d/manifest.json"; [ -f "$MF" ] || continue
  rel="${d#"$SRC_ROOT/"}"; rel="${rel%/*.chunks}"
  out_dir="$DEST_ROOT/$rel"; mkdir -p "$out_dir"
  OUT_BASENAME="$("$PY" - <<'PY' "$MF"
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("original_filename","restored.bin"))
PY
)"
  "$PY" "$JOIN" "$d" --out "$out_dir/$OUT_BASENAME" --passphrase "$PASSPHRASE"
done

echo "DONE: Assembled + copied to $DEST_ROOT"
