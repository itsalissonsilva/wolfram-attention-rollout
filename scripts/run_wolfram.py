"""Run a Wolfram script in a hidden kernel and retain its execution record."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--reuse-weight-audit", type=Path,
                        help="Reuse completed parameter hashes in a prior native audit; forward tests rerun.")
    parser.add_argument("--kernel", default=os.environ.get(
        "WOLFRAM_KERNEL", r"C:\Program Files\Wolfram Research\Wolfram\15.0\SystemFiles\Kernel\Binaries\Windows-x86-64\WolframKernel.exe"))
    args = parser.parse_args()
    script = args.script.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "launch.json").exists():
        parser.error("Use a new output directory to preserve prior execution records.")
    environment = os.environ.copy()
    environment["ATTENTION_ROLLOUT_OUTPUT"] = str(output)
    if args.reuse_weight_audit:
        environment["ATTENTION_ROLLOUT_WEIGHT_AUDIT"] = str(args.reuse_weight_audit.resolve(strict=True))
    command = [args.kernel, "-noinit", "-noprompt", "-run",
               "Get[" + json.dumps(script.as_posix()) + "]; Quit[]"]
    started = time.monotonic()
    record = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script": str(script), "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    (output / "launch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (output / "stdout.txt").open("wb") as stdout, (output / "stderr.txt").open("wb") as stderr:
        try:
            result = subprocess.run(command, cwd=script.parent, env=environment,
                                    stdout=stdout, stderr=stderr, timeout=args.timeout,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            record["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            record.update(returncode=None, timeout=True)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result_path = output / "result.json"
    record["result_written"] = result_path.is_file()
    if result_path.is_file():
        record["success"] = json.loads(result_path.read_text(encoding="utf-8-sig")).get("Success", False)
    (output / "launch.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    # Windows kernels may not emit stdout. Scripts must export their own results.
    if record.get("returncode") != 0 or not record.get("success", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
