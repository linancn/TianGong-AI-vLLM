import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "runtime_patches",
    Path(__file__).resolve().parents[1] / "deploy/vllm/patches/apply_runtime_fixes.py",
)
patches = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patches)


def test_patch_checks_upstream_identity_and_compiles_result(tmp_path):
    path = tmp_path / "upstream.py"
    source = "value = 1\n"
    path.write_text(source)
    digest = hashlib.sha256(source.encode()).hexdigest()
    patches.replace_checked(path, digest, "value = 1", "value = 2")
    assert path.read_text() == "value = 2\n"
    with pytest.raises(RuntimeError, match="Upstream file changed"):
        patches.replace_checked(path, digest, "value = 2", "value = 3")
    assert path.read_text() == "value = 2\n"


@pytest.mark.parametrize("before,after", [("absent", "value = 2"), ("value = 1", "invalid(")])
def test_bad_patch_leaves_upstream_untouched(tmp_path, before, after):
    path = tmp_path / "upstream.py"
    source = "value = 1\n"
    path.write_text(source)
    with pytest.raises((RuntimeError, SyntaxError)):
        patches.replace_checked(path, hashlib.sha256(source.encode()).hexdigest(), before, after)
    assert path.read_text() == source
