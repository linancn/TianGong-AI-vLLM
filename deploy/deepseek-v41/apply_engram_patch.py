"""Apply the reviewed MXFP4 Engram backport to the exact vLLM source."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess

spec = importlib.util.find_spec("vllm")
assert spec and spec.origin
target = Path(spec.origin).parent / "models/deepseek_v41/common/engram.py"
expected = "720923ccd796dfdeab4b396eeeea79cd4c11a9dcd2e6ef8d57c0390fa4cf7ab2"
assert (
    hashlib.sha256(target.read_bytes()).hexdigest() == expected
), "vLLM Engram source changed; review the backport before building"
subprocess.run(
    ["patch", "--batch", "--fuzz=0", str(target), "/opt/tiangong/engram-mxfp4.patch"],
    check=True,
)
compile(target.read_text(), str(target), "exec")
print("Pinned MXFP4 Engram backport applied")
