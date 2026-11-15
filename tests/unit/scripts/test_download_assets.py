# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise asset cache behavior without network access or native libraries."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/download_assets.sh"


@pytest.fixture(params=["wget", "curl"])
def asset_download(tmp_path, request):
    root = tmp_path / "vision checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    configs = root / "examples/sample/config"
    configs.mkdir(parents=True)
    (configs / "sample.yaml").write_text(
        "test_image: ~/.cache/assets/image/sample.jpg\n", encoding="utf-8"
    )
    (root / "applications").mkdir()

    # Expose only the selected downloader, even if both are installed locally.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in (
        "dirname", "mkdir", "find", "grep", "sort", "mktemp", "mv", "rm", "ls"
    ):
        executable = shutil.which(command)
        assert executable is not None, f"Required command not found: {command}"
        (bin_dir / command).symlink_to(executable)
    downloader = bin_dir / request.param
    downloader.write_text(
        "#!/bin/sh\n"
        'printf "call\\n" >> "$DOWNLOAD_CALLS"\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in\n'
        '    -O|-o) destination="$2"; shift 2 ;;\n'
        '    *) shift ;;\n'
        '  esac\n'
        'done\n'
        'if [ "$DOWNLOAD_FAIL" = 1 ]; then\n'
        '  printf "partial" > "$destination"\n'
        '  exit 4\n'
        'fi\n'
        'printf "complete asset" > "$destination"\n',
        encoding="utf-8",
    )
    downloader.chmod(0o755)
    home = tmp_path / "test home"
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": str(bin_dir),
        "DOWNLOAD_CALLS": str(calls),
        "VISION_ASSETS_URL": "https://assets.invalid",
    }

    def run(fail=False):
        return subprocess.run(
            ["/bin/sh", str(script)],
            env={**env, "DOWNLOAD_FAIL": "1" if fail else "0"},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    return run, home / ".cache/assets/image/sample.jpg", calls


def test_failed_download_is_not_cached_and_can_be_retried(asset_download):
    run, asset, calls = asset_download
    failed = run(fail=True)
    assert failed.returncode != 0
    assert not asset.exists(), "A partial download must not become a cached asset"
    assert list(asset.parent.iterdir()) == [], "Partial files must be cleaned up"

    retried = run()
    assert retried.returncode == 0, retried.stderr
    assert asset.read_bytes() == b"complete asset"
    assert calls.read_text().splitlines() == ["call", "call"]
    assert list(asset.parent.iterdir()) == [asset]


def test_existing_asset_is_reused_without_downloading(asset_download):
    run, asset, calls = asset_download
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"existing asset")

    result = run(fail=True)
    assert result.returncode == 0, result.stderr
    assert asset.read_bytes() == b"existing asset"
    assert not calls.exists()
