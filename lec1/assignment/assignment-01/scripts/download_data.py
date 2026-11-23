"""Re-download the real dataset if data/real/tinyshakespeare.txt is missing.

    python scripts/download_data.py
"""
import hashlib
import pathlib
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DEST = pathlib.Path(__file__).resolve().parents[1] / "data" / "real" / "tinyshakespeare.txt"

if DEST.exists():
    print("already present:", DEST)
else:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        print("warning: checksum differs from the reference copy (results may differ slightly)")
    DEST.write_bytes(data)
    print("downloaded", len(data), "bytes ->", DEST)
