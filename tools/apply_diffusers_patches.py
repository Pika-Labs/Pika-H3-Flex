#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Apply the exact tested H3 fixes to the pinned installed Diffusers package."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify patched files without changing anything")
    parser.add_argument("--root", type=Path, help="Diffusers package directory (normally discovered automatically)")
    args = parser.parse_args()
    patches = Path(__file__).resolve().parent.parent / "patches"
    manifest = json.loads((patches / "manifest.json").read_text())
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("diffusers")
        if spec is None or spec.origin is None:
            raise RuntimeError("Install requirements.txt before applying Diffusers patches")
        root = Path(spec.origin).parent
    root = root.resolve()
    pending = []
    for item in manifest["patches"]:
        patch = patches / item["file"]
        if sha256(patch) != item["sha256"]:
            raise RuntimeError(f"Patch checksum mismatch: {patch}")
        target = root / item["target"]
        digest = sha256(target)
        if digest == item["after_sha256"]:
            continue
        if args.check or digest not in [item["before_sha256"]] + [u["sha256"] for u in item.get("notice_upgrades", [])]:
            raise RuntimeError(f"Unexpected or unpatched Diffusers file: {target} (sha256={digest}); "
                               f"expected tested patches on upstream {manifest['upstream_revision']}")
        pending.append(item)
    if pending:
        if shutil.which("patch") is None:
            raise RuntimeError("Install the system 'patch' executable first")
        # Validate every source before changing anything; stage and verify all outputs.
        with tempfile.TemporaryDirectory(prefix="pika-h3-diffusers-") as temporary:
            stage = Path(temporary)
            for item in pending:
                staged = stage / item["target"]
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / item["target"], staged)
                upgrade = next((u for u in item.get("notice_upgrades", []) if sha256(staged) == u["sha256"]), None)
                if upgrade is not None:
                    payload = staged.read_bytes()
                    old_header = upgrade["remove_header"].encode("utf-8")
                    if not payload.startswith(old_header):
                        raise RuntimeError("Previous modification header mismatch")
                    staged.write_bytes(item["modification_notice_header"].encode("utf-8") + payload[len(old_header):])
                else:
                    subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p3", "-i",
                                    str(patches / item["file"])], cwd=stage, check=True)
                if sha256(staged) != item["after_sha256"]:
                    raise RuntimeError(f"Patched checksum mismatch: {item['target']}")
            for item in pending:
                target = root / item["target"]
                # Atomic replacement also changes mtime, invalidating cached bytecode.
                fd, name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write((stage / item["target"]).read_bytes())
                    os.chmod(name, target.stat().st_mode & 0o777)
                    os.replace(name, target)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
    for item in manifest["patches"]:
        if sha256(root / item["target"]) != item["after_sha256"]:
            raise RuntimeError(f"Final verification failed: {item['target']}")
    print(f"Diffusers H3 patches verified ({len(pending)} applied): {root}")


if __name__ == "__main__":
    main()
