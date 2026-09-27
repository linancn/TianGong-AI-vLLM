"""Protect the three-GPU Embed lifecycle from colliding with the four-GPU service."""

from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_manage(tmp_path: Path, group: str, running_project: str = "") -> tuple[int, list[str]]:
    (tmp_path / "deploy").mkdir()
    (tmp_path / "deploy/manage.sh").write_bytes((ROOT / "deploy/manage.sh").read_bytes())
    (tmp_path / ".env").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$DOCKER_LOG"\n'
        'if [ "$1" = ps ] && [ -n "$RUNNING_PROJECT" ]; then\n'
        '  case " $* " in *"label=com.docker.compose.project=$RUNNING_PROJECT"*) '
        "echo running-container;; esac\n"
        "fi\n"
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "DOCKER_LOG": str(log),
        "RUNNING_PROJECT": running_project,
    }
    result = subprocess.run(
        ["bash", str(tmp_path / "deploy/manage.sh"), "start", group],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, log.read_text().splitlines()


@pytest.mark.parametrize(
    ("group", "other_project"),
    [("embed3", "tiangong-embed"), ("embed", "tiangong-embed3")],
)
def test_embed_profiles_cannot_start_together(tmp_path, group, other_project):
    status, calls = run_manage(tmp_path, group, other_project)
    assert status == 2
    assert len(calls) == 1
    assert calls[0].startswith("ps --quiet --filter")


def test_embed3_start_uses_isolated_compose_without_build_or_pull(tmp_path):
    status, calls = run_manage(tmp_path, "embed3")
    assert status == 0
    assert len(calls) == 2
    assert "/deploy/embed3/compose.yaml" in calls[1]
    assert "up -d --no-build --pull never --no-recreate embed" in calls[1]


def test_embed3_start_includes_private_host_override(tmp_path):
    override = tmp_path / "output/instances/embed3.compose.yaml"
    override.parent.mkdir(parents=True)
    override.write_text("services: {}\n")
    status, calls = run_manage(tmp_path, "embed3")
    assert status == 0
    assert f"-f {tmp_path}/deploy/embed3/compose.yaml -f {override}" in calls[1]
