"""Download V-JEPA 2 checkpoints (download only - no model is loaded or run).

Examples
--------
    # the action-conditioned checkpoint (default), into ./checkpoints
    python scripts/download_checkpoints.py

    # also grab the ViT-L encoder-only checkpoint from HuggingFace
    python scripts/download_checkpoints.py --encoder vitl

    # only an encoder, skip the (large) AC checkpoint
    python scripts/download_checkpoints.py --encoder vitg --no-ac

Verified sources (primary):
- AC checkpoint (encoder + predictor state dicts), trained from ViT-g on ~62 h
  DROID: https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
- Encoder-only checkpoints on HuggingFace (MIT / Apache-2.0):
  facebook/vjepa2-{vitl,vith,vitg}-fpc64-256, facebook/vjepa2-vitg-fpc64-384
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

AC_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
ENCODERS = {
    "vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vith": "facebook/vjepa2-vith-fpc64-256",
    "vitg": "facebook/vjepa2-vitg-fpc64-256",
    "vitg384": "facebook/vjepa2-vitg-fpc64-384",
}


def _human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def download_stream(url: str, dest: str, chunk: int = 1 << 20) -> str:
    """Resumable streamed download with a simple progress line."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  resuming from {_human(existing)}")
    try:
        resp = urllib.request.urlopen(req)  # noqa: S310 - trusted Meta/HF hosts
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:
            # Range not satisfiable: the .part is already the complete file. Finalize it.
            os.replace(tmp, dest)
            print("  already complete; finalized")
            return dest
        raise
    with resp:
        total = int(resp.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing and resp.status == 206 else "wb"
        if mode == "wb":
            existing = 0
        done = existing
        with open(tmp, mode) as out:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(f"\r  {_human(done)} / {_human(total)} ({pct:5.1f}%)")
                else:
                    sys.stdout.write(f"\r  {_human(done)}")
                sys.stdout.flush()
    sys.stdout.write("\n")
    if total and done != total:
        raise IOError(f"incomplete download: got {done} of {total} bytes; kept {tmp} for resume")
    os.replace(tmp, dest)
    return dest


def download_encoder(model_id: str, dest_dir: str) -> str:
    from huggingface_hub import snapshot_download

    target = os.path.join(dest_dir, model_id.split("/")[-1])
    print(f"[encoder] {model_id} -> {target}")
    snapshot_download(repo_id=model_id, local_dir=target,
                      allow_patterns=["*.safetensors", "*.json", "*.txt"])
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="checkpoints", help="output directory")
    parser.add_argument("--encoder", choices=list(ENCODERS) + ["none"], default="none",
                        help="also download a HuggingFace encoder-only checkpoint")
    parser.add_argument("--ac", dest="ac", action="store_true", default=True,
                        help="download the action-conditioned checkpoint (default)")
    parser.add_argument("--no-ac", dest="ac", action="store_false",
                        help="skip the action-conditioned checkpoint")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    if args.ac:
        dest = os.path.join(args.dest, os.path.basename(AC_URL))
        if os.path.exists(dest):
            print(f"[ac] already present: {dest} ({_human(os.path.getsize(dest))})")
        else:
            print(f"[ac] {AC_URL}")
            download_stream(AC_URL, dest)
            print(f"[ac] saved {dest} ({_human(os.path.getsize(dest))})")

    if args.encoder != "none":
        download_encoder(ENCODERS[args.encoder], args.dest)

    print("done.")


if __name__ == "__main__":
    main()
