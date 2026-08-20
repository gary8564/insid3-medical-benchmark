from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.constants import INSID3_N_EPISODES, PREVIEW_N
from src.run_insid3 import main
from tests.conftest import write_domain_cache

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_polyp_cache(tmp_path: Path, ids: list[str] | None = None) -> Path:
    data_root = tmp_path / "processed"
    write_domain_cache(data_root, "polyp", ids or ["ref", "t1", "t2"])
    return data_root


def _episodes_json(output_dir: Path) -> Path:
    return output_dir / "episodes.json"


def test_dry_run_five_shot_lists_all_references(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path, ["a", "b", "c", "d", "e", "f"])
    output_dir = tmp_path / "out"
    code = main(
        [
            "--dataset",
            "polyp",
            "--input-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--shots",
            "5",
            "--episodes",
            "4",
        ]
    )
    assert code == 0
    payload = json.loads(_episodes_json(output_dir).read_text())
    assert len(payload) == 4
    for row in payload:
        assert "reference_id" not in row
        assert len(row["reference_ids"]) == 5
        assert row["target_id"] not in row["reference_ids"]
        assert len(row["reference_images"]) == 5
        assert all(Path(path).is_file() for path in row["reference_images"])


def test_flexict2d_rejects_polyp(tmp_path: Path):
    from src.methods.insid3.run import load_episodes, parse_args

    args = parse_args(
        [
            "--dataset",
            "polyp",
            "--backbone",
            "flexict2d",
            "--episodes-json",
            str(tmp_path / "episodes.json"),
        ]
    )
    with pytest.raises(SystemExit, match="out of scope"):
        load_episodes(args)


def test_dry_run_builds_episodes_without_loading_insid3(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path)
    output_dir = tmp_path / "out"

    code = main(
        [
            "--dataset",
            "polyp",
            "--input-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 0
    payload = json.loads(_episodes_json(output_dir).read_text())
    assert len(payload) == INSID3_N_EPISODES
    assert all(row["reference_ids"][0] != row["target_id"] for row in payload)
    assert {row["reference_ids"][0] for row in payload} != {"ref"}
    assert payload[0]["dataset"] == "polyp"
    assert payload[0]["episode_index"] == 0
    for row in payload:
        assert len(row["reference_ids"]) == 1
        assert "reference_id" not in row
        assert Path(row["reference_images"][0]).is_file()
        assert Path(row["reference_masks"][0]).is_file()
        assert Path(row["target_image"]).is_file()


def test_dry_run_cli_subprocess(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path)
    output_dir = tmp_path / "out"
    script = REPO_ROOT / "src" / "run_insid3.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "polyp",
            "--input-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    payload = json.loads(result.stdout)
    assert len(payload) == INSID3_N_EPISODES
    assert json.loads(_episodes_json(output_dir).read_text()) == payload
    assert all(row["reference_ids"][0] != row["target_id"] for row in payload)


def test_dry_run_does_not_import_torch(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path)
    output_dir = tmp_path / "out"
    argv = [
        "--dataset",
        "polyp",
        "--input-dir",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.run_insid3 import main\n"
        "assert 'torch' not in sys.modules\n"
        f"raise SystemExit(main({argv!r}))\n"
    )
    nested = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert nested.returncode == 0, nested.stderr
    assert len(json.loads(nested.stdout)) == INSID3_N_EPISODES


def test_preview_without_n_uses_constant_and_limits_dry_run(tmp_path: Path):
    data_root = _write_polyp_cache(
        tmp_path, ["ref"] + [f"t{i}" for i in range(PREVIEW_N + 3)]
    )
    output_dir = tmp_path / "out"
    code = main(
        [
            "--dataset",
            "polyp",
            "--input-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--preview",
        ]
    )
    assert code == 0
    payload = json.loads(_episodes_json(output_dir).read_text())
    assert len(payload) == PREVIEW_N


def test_preview_n_overrides_constant(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path)
    output_dir = tmp_path / "out"
    code = main(
        [
            "--dataset",
            "polyp",
            "--input-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--preview",
            "1",
        ]
    )
    assert code == 0
    assert len(json.loads(_episodes_json(output_dir).read_text())) == 1


def test_episodes_and_seed_are_reproducible(tmp_path: Path):
    data_root = _write_polyp_cache(tmp_path)
    argv = [
        "--dataset",
        "polyp",
        "--input-dir",
        str(data_root),
        "--dry-run",
        "--episodes",
        "7",
        "--seed",
        "0",
    ]
    first = tmp_path / "a"
    second = tmp_path / "b"
    assert main(argv + ["--output-dir", str(first)]) == 0
    assert main(argv + ["--output-dir", str(second)]) == 0
    assert json.loads(_episodes_json(first).read_text()) == json.loads(
        _episodes_json(second).read_text()
    )
    other = tmp_path / "c"
    assert main(argv[:-1] + ["1", "--output-dir", str(other)]) == 0
    assert json.loads(_episodes_json(other).read_text()) != json.loads(
        _episodes_json(first).read_text()
    )
