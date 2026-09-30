# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `dora.toml` project settings."""

from pathlib import Path

import pytest

from .. import project


def write(folder: Path, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / project.TOML_NAME
    path.write_text(text)
    return path


def test_found_by_walking_up(tmp_path):
    """`dora` should work from a subdirectory, which the package scan cannot do."""
    write(tmp_path, '[project]\npackage = "proj"\n')
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert project.find_toml(deep) == tmp_path / project.TOML_NAME


def test_absent_is_not_an_error(tmp_path):
    # tmp_path has no dora.toml, and neither does anything above /tmp.
    assert project.find_toml(tmp_path) is None
    assert project.load(tmp_path) is None


def test_reads_project_and_dora_sections(tmp_path, monkeypatch):
    monkeypatch.setenv("XP_ROOT", "/tmp/xps")
    write(
        tmp_path,
        """
[project]
package = "proj"
main_module = "trainer"
config_path = "conf"
config_name = "main"
hydra = { version_base = "1.1" }

[dora]
dir = "${env:XP_ROOT}"
exclude = ["device", "wandb.*"]
git_save = true
""",
    )
    conf = project.load(tmp_path)
    assert conf is not None
    assert (conf.package, conf.main_module) == ("proj", "trainer")
    assert conf.hydra_kwargs == {"version_base": "1.1"}
    dora = conf.dora_config()
    assert dora is not None
    assert dora.dir == Path("/tmp/xps")
    assert dora.git_save is True
    assert dora.is_excluded("wandb.project")


def test_env_fallback_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("XP_ROOT", raising=False)
    write(tmp_path, '[dora]\ndir = "${env:XP_ROOT:-/tmp/fallback}"\n')
    dora = project.load(tmp_path).dora_config()
    assert dora is not None and dora.dir == Path("/tmp/fallback")


def test_unresolvable_dir_yields_no_config(tmp_path, monkeypatch):
    """Better to make the caller import the project than to point at the wrong
    experiment directory, which would silently appear empty."""
    monkeypatch.delenv("XP_ROOT", raising=False)
    write(tmp_path, '[dora]\ndir = "${env:XP_ROOT}"\n')
    assert project.load(tmp_path).dora_config() is None


def test_relative_dir_is_anchored_to_the_repository(tmp_path):
    write(tmp_path, '[dora]\ndir = "outputs"\n')
    dora = project.load(tmp_path).dora_config()
    assert dora is not None and dora.dir == (tmp_path / "outputs").resolve()


def test_unknown_dora_key_is_rejected(tmp_path):
    write(tmp_path, '[dora]\ndir = "/tmp/x"\nnot_a_field = 1\n')
    with pytest.raises(project.ProjectConfigError, match="not_a_field"):
        project.load(tmp_path).dora_config()


def test_dir_probe_picks_the_first_existing_marker(tmp_path, monkeypatch):
    """Projects that run on several clusters choose their experiment directory
    by looking for a marker path; the probe list says so statically."""
    monkeypatch.delenv("XP_ROOT", raising=False)
    monkeypatch.setenv("USER", "someone")
    marker = tmp_path / "cluster_b"
    marker.mkdir()
    write(
        tmp_path,
        f'''
[dora]
dir = "${{env:XP_ROOT}}"

[[dora.dir_probe]]
probe = "{tmp_path / "cluster_a"}"
dir   = "/xps/a/${{env:USER}}"

[[dora.dir_probe]]
probe = "{marker}"
dir   = "/xps/b/${{env:USER}}"
''',
    )
    dora = project.load(tmp_path).dora_config()
    assert dora is not None and dora.dir == Path("/xps/b/someone")


def test_dir_wins_over_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("XP_ROOT", "/xps/explicit")
    marker = tmp_path / "marker"
    marker.mkdir()
    write(
        tmp_path,
        f'''
[dora]
dir = "${{env:XP_ROOT}}"

[[dora.dir_probe]]
probe = "{marker}"
dir   = "/xps/probed"
''',
    )
    assert project.load(tmp_path).dora_config().dir == Path("/xps/explicit")


def test_no_probe_matches_yields_no_config(tmp_path, monkeypatch):
    monkeypatch.delenv("XP_ROOT", raising=False)
    write(
        tmp_path,
        f'''
[dora]
dir = "${{env:XP_ROOT}}"

[[dora.dir_probe]]
probe = "{tmp_path / "absent"}"
dir   = "/xps/a"
''',
    )
    assert project.load(tmp_path).dora_config() is None


def test_malformed_probe_is_rejected(tmp_path):
    write(tmp_path, '[dora]\ndir = "${env:NOPE}"\n\n[[dora.dir_probe]]\nprobe = "/tmp"\n')
    with pytest.raises(project.ProjectConfigError, match="dir_probe"):
        project.load(tmp_path).dora_config()


def test_dora_config_without_dir_still_carries_exclusions(tmp_path, monkeypatch):
    """`exclude` decides signatures, so it must survive even when the
    experiment directory cannot be resolved -- the training path gets its
    directory elsewhere, but silently dropping the exclusions would re-sign
    every experiment in the project."""
    monkeypatch.delenv("XP_ROOT", raising=False)
    write(tmp_path, '[dora]\ndir = "${env:XP_ROOT}"\nexclude = ["device"]\n')
    conf = project.load(tmp_path)

    assert conf.dora_config() is None  # read-only path: refuse to guess
    lenient = conf.dora_config(require_dir=False)  # training path: keep what we know
    assert lenient is not None
    assert lenient.is_excluded("device")


def test_no_dora_section_yields_no_config(tmp_path):
    write(tmp_path, '[project]\npackage = "proj"\n')
    assert project.load(tmp_path).dora_config(require_dir=False) is None


@pytest.mark.parametrize("setting, expected", [("", False), ("false", False), ("true", True)])
def test_fast_parser_flag(tmp_path, setting, expected):
    option = f"use_fast_parser = {setting}\n" if setting else ""
    write(tmp_path, "[project]\n" + option)
    assert project.load(tmp_path).use_fast_parser is expected


@pytest.mark.parametrize("value", ['"true"', '"false"', "1", "[]"])
def test_fast_parser_flag_requires_boolean(tmp_path, value):
    write(tmp_path, f"[project]\nuse_fast_parser = {value}\n")
    with pytest.raises(project.ProjectConfigError, match="use_fast_parser must be a boolean"):
        project.load(tmp_path)


def test_name_width_without_experiment_directory(tmp_path, monkeypatch):
    from dora.conf import DoraConfig
    from dora.inspect import get_name_width

    monkeypatch.chdir(tmp_path)
    assert get_name_width() == 100
    assert get_name_width(DoraConfig(name_width=64)) == 64
    write(tmp_path, "[dora]\nname_width = 32\n")
    assert get_name_width(DoraConfig(name_width=64)) == 32
    assert project.load(tmp_path).dora_config(require_dir=False).name_width == 32


@pytest.mark.parametrize("value", ["0", "-1", "true", '"80"', "2.5"])
def test_invalid_name_width(tmp_path, value):
    write(tmp_path, f"[dora]\nname_width = {value}\n")
    with pytest.raises(ValueError, match="name_width must be a positive integer"):
        project.load(tmp_path).dora_config(require_dir=False)
