#!/usr/bin/env python3
"""Run pip check, accounting only for decord 0.6.0's mislabeled Linux wheel."""
import importlib
from importlib.metadata import distribution
import platform
import subprocess
import sys


def main():
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    if not result.returncode:
        return
    if (result.stdout.strip() != "decord 0.6.0 is not supported on this platform"
            or platform.system() != "Linux" or platform.machine() != "x86_64"):
        raise SystemExit(result.returncode)
    package = distribution("decord")
    if (package.version != "0.6.0"
            or "Tag: cp36-cp36m-manylinux2010_x86_64" not in (package.read_text("WHEEL") or "")):
        raise SystemExit(result.returncode)
    # Its published py3 wheel embeds a stale CPython 3.6 metadata tag. Do not
    # mask import/native-library failures or unrelated pip check diagnostics.
    decord = importlib.import_module("decord")
    if decord.__version__ != "0.6.0":
        raise SystemExit("Unexpected decord module version")
    print("COMPAT: decord 0.6.0 imported successfully; only its known cp36 wheel metadata tag is exempted.")


if __name__ == "__main__":
    main()
