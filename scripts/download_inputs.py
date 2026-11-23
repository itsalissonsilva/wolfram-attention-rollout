"""Download the pinned source photographs and, optionally, the Wolfram model."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CASES = ("samoyed", "samoyed-option-4", "samoyed-option-5", "samoyed-option-1")


def fetch(item):
    path = ROOT / item["LocalFile"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(item["URL"], headers={"User-Agent": "WolframAttentionComparison/1.0"})
        partial = path.with_suffix(path.suffix + ".part")
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > item["Bytes"]:
                    raise ValueError(f"Unexpected download size: {path.name}")
                digest.update(chunk)
                output.write(chunk)
        if count != item["Bytes"] or digest.hexdigest() != item["SHA256"]:
            raise ValueError(f"Downloaded file differs from the recorded input: {path.name}")
        partial.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["SHA256"]:
        raise ValueError(f"Cached file differs from the recorded input: {path.name}")
    print(f"Verified {path.relative_to(ROOT).as_posix()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", choices=CASES, help="Download only this photograph; default: all four.")
    parser.add_argument("--model", action="store_true", help="Also download the 346 MB ViT Base/16 model.")
    args = parser.parse_args()
    for case in (args.image,) if args.image else CASES:
        fetch(json.loads((ROOT / "experiments" / f"{case}.json").read_text(encoding="utf-8")))
    if args.model:
        fetch(json.loads((ROOT / "experiments/model.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
