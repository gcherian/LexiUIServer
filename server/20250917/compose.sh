#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: ./compose.sh <hf_model_id>}"
REPO_URL="${REPO_URL:-https://github.com/exchange.git}"
SAFE_NAME="$(echo "$MODEL_ID" | tr '/:' '__')"
WORKDIR="exchange-tmp"
VENV=".ml-venv"

read -rs -p "Passphrase: " PASSPHRASE; echo

command -v git >/dev/null
command -v python3 >/dev/null

[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
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

mkdir -p "$WORKDIR/models/$SAFE_NAME" tools
DL_TMP="$(python3 - <<'PY'
import tempfile; print(tempfile.mkdtemp(prefix="hf-snap-"))
PY
)"
python3 - "$MODEL_ID" "$DL_TMP" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], ignore_patterns=["*.md",".gitattributes",".git/*","**/.git/*"])
print(sys.argv[2])
PY

python3 - "$DL_TMP" "$WORKDIR/models/$SAFE_NAME" "$PASSPHRASE" <<'PY'
import os, sys, json, hashlib, base64, secrets, shutil
from pathlib import Path
SRC=Path(sys.argv[1]); DEST=Path(sys.argv[2]); PW=sys.argv[3].encode()
BUF=1024*1024
def hb(s):
    s=s.lower()
    return int(float(s[:-1])*1024**({"k":1,"m":2,"g":3}[s[-1]])) if s[-1] in "kmg" else int(s)
CHUNK=hb("50m"); THR=hb("25m")
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(BUF), b""): h.update(b)
    return h.hexdigest()
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
def kdf(pw,salt,rounds=200_000):
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=rounds).derive(pw)
DEST.mkdir(parents=True, exist_ok=True)
files=[p for p in SRC.rglob("*") if p.is_file()]
for fp in files:
    rel=fp.relative_to(SRC); outp=(DEST/rel.parent); outp.mkdir(parents=True, exist_ok=True)
    sz=fp.stat().st_size
    if sz<=THR:
        shutil.copy2(fp, outp/fp.name); continue
    base=fp.name; chunk_dir=outp/f"{base}.chunks"; chunk_dir.mkdir(parents=True, exist_ok=True)
    salt=os.urandom(16); key=kdf(PW, salt)
    man={"original_filename":base,"original_size":sz,"original_sha256":sha(fp),"chunk_size":CHUNK,"chunks":[],
         "encrypted":True,"kdf":{"algo":"PBKDF2-HMAC-SHA256","rounds":200000},"salt_b64":base64.b64encode(salt).decode()}
    with open(fp,"rb") as f:
        i=0
        while True:
            buf=f.read(CHUNK)
            if not buf: break
            nonce=os.urandom(12); ct=AESGCM(key).encrypt(nonce, buf, base.encode())
            part=f"{base}.part{i:04d}"; (chunk_dir/part).write_bytes(nonce+ct)
            man["chunks"].append({"name":part,"size":len(ct),"nonce_b64":base64.b64encode(nonce).decode(),"sha256":hashlib.sha256(ct).hexdigest()})
            i+=1
    (chunk_dir/"manifest.json").write_text(json.dumps(man,indent=2),encoding="utf-8")
tools=Path("tools"); tools.mkdir(exist_ok=True)
jp=tools/"join_file.py"
if not jp.exists():
    jp.write_text(r'''#!/usr/bin/env python3
import argparse, base64, hashlib, json, pathlib, sys, getpass
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
BUF=1024*1024
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(BUF), b""): h.update(b)
    return h.hexdigest()
def derive(pw,salt,rounds):
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=rounds).derive(pw)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("chunks_dir"); ap.add_argument("--out",default=None); ap.add_argument("--passphrase",default=None)
    a=ap.parse_args()
    d=pathlib.Path(a.chunks_dir); man=json.loads((d/"manifest.json").read_text())
    out=pathlib.Path(a.out or man["original_filename"])
    pw=(a.passphrase or getpass.getpass("Passphrase: ")).encode()
    salt=base64.b64decode(man["salt_b64"]); rounds=int(man["kdf"]["rounds"])
    key=derive(pw,salt,rounds)
    with open(out,"wb") as w:
        for ch in man["chunks"]:
            raw=(d/ch["name"]).read_bytes(); nonce,ct=raw[:12],raw[12:]
            if hashlib.sha256(ct).hexdigest()!=ch["sha256"]:
                print("Hash mismatch:", ch["name"], file=sys.stderr); sys.exit(3)
            w.write(AESGCM(key).decrypt(nonce, ct, man["original_filename"].encode()))
    ok=out.stat().st_size==man["original_size"] and sha(out)==man["original_sha256"]
    print("OK" if ok else "FAIL")
if __name__=="__main__": main()
''', encoding="utf-8")
PY

cd "$WORKDIR"
git fetch origin || true
if ! git rev-parse --verify main >/dev/null 2>&1; then
  git checkout -b main
else
  git checkout main
  git pull --ff-only || true
fi

printf "%s\n%s\n%s\n%s\n%s\n%s\n%s\n" ".ml-venv/" "__pycache__/" "*.cache/" "*.tmp" ".DS_Store" "hf-snap-*/" ".parts.*" > .gitignore
git add .gitignore || true
git commit -m "init ignore" || true
git push origin main || true

git add tools/join_file.py || true
git commit -m "add join tool" || true
git push origin main || true

find "models/$SAFE_NAME" -type f -name "*.part*" | sort | while read -r f; do
  git add "$f"
  git commit -m "$SAFE_NAME: add $(basename "$f")" || true
  git push origin main
done

find "models/$SAFE_NAME" -type f -name "manifest.json" -o -name "*.json" ! -name "*.part*" | sort | while read -r f; do
  git add "$f"
done
git commit -m "$SAFE_NAME: add manifests & metadata" || true
git push origin main

echo "DONE: models/$SAFE_NAME pushed to $REPO_URL on branch main"
