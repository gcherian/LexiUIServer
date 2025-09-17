#!/usr/bin/env python3
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
    d=pathlib.Path(a.chunks_dir)
    man=json.loads((d/"manifest.json").read_text(encoding="utf-8"))
    out=pathlib.Path(a.out or man["original_filename"])
    pw=(a.passphrase or getpass.getpass("Passphrase: ")).encode()
    salt=base64.b64decode(man["salt_b64"]); rounds=int(man["kdf"]["rounds"])
    key=derive(pw,salt,rounds)
    with open(out,"wb") as w:
        for ch in man["chunks"]:
            raw=(d/ch["name"]).read_bytes(); nonce,ct=raw[:12], raw[12:]
            if hashlib.sha256(ct).hexdigest()!=ch["sha256"]:
                print(f"Hash mismatch: {ch['name']}", file=sys.stderr); sys.exit(3)
            w.write(AESGCM(key).decrypt(nonce, ct, man["original_filename"].encode()))
    ok = out.stat().st_size==man["original_size"] and sha(out)==man["original_sha256"]
    if not ok: print("Integrity check failed.", file=sys.stderr); sys.exit(4)
    print(f"OK → {out} ({out.stat().st_size} bytes)")
if __name__=="__main__": main()
