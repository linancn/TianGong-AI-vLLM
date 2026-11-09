"""Backport vLLM #65356 to the exact 0.33.0 SM120 backend declarations."""

import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
assert spec and spec.origin
root = Path(spec.origin).parent
changes = [
    (
        "models/deepseek_v41/nvidia/flashinfer_sparse.py",
        "7926b4e3fa44fe0fc1ed3f93f0f292b4f56fa2103cf93c30a94d93958f84d682",
        [
            (
                "from vllm.platforms.interface import DeviceCapability",
                "from vllm.platforms import current_platform\n"
                "from vllm.platforms.interface import DeviceCapability",
            ),
            (
                "        return [128]\n",
                "        return [64 if current_platform.is_device_capability_family(120) else 128]\n",
            ),
        ],
    ),
    (
        "v1/attention/backends/mla/indexer.py",
        "6457fb5a755ca53addce7731cb56770493371ad9f3728251e7ff33515b67965c",
        [
            (
                "        return [64 if current_platform.is_device_capability_family(90) else 128]\n",
                "        return [\n"
                "            64\n"
                "            if (current_platform.is_device_capability_family(90)\n"
                "                or current_platform.is_device_capability_family(120))\n"
                "            else 128\n"
                "        ]\n",
            )
        ],
    ),
]
for relative, digest, replacements in changes:
    target = root / relative
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, relative
    content = target.read_text()
    for before, after in replacements:
        assert content.count(before) == 1, (relative, "Patch context changed")
        content = content.replace(before, after)
    compile(content, str(target), "exec")
    target.write_text(content)
print("Pinned SM120 64-token page declarations applied")
