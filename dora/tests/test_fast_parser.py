"""Exercise the actual opt-in decorator, including task execution."""

import json
import logging
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from dora import hydra_main, to_absolute_path, get_xp
from dora.parser import UnsupportedFeature


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "defaults: [_self_, {solver: small}]\n"
        "lr: 0.01\noptim: {momentum: 0.9}\nvalues: [1, 2]\nalias: ${lr}\n"
        "num_workers: 1\nslurm: {gpus: 2}\n"
    )
    (tmp_path / "solver").mkdir()
    for name, width in (("small", 8), ("big", 16)):
        (tmp_path / "solver" / (name + ".yaml")).write_text(
            f"# @package _global_\nwidth: {width}\n"
        )
    return tmp_path


def make_main(root, flag="true", **kwargs):
    setting = f"use_fast_parser = {flag}\n" if flag else ""
    (root / "dora.toml").write_text(
        "[project]\n" + setting + '[dora]\ndir = "outputs"\nexclude = ["num_workers"]\n'
    )

    def task(cfg):
        assert cfg is get_xp().cfg
        logging.getLogger("training").info("running task")
        return get_xp(), Path.cwd(), to_absolute_path("data")

    return hydra_main(config_path=str(root), config_name="config", **kwargs)(task)


@pytest.mark.parametrize("flag", ["false", ""])
def test_hydra_stays_default(config, flag):
    main = make_main(config, flag, version_base="1.1")
    assert main.use_fast_parser is False
    assert not hasattr(main, "_parser")
    assert main.get_xp(["solver=big"]).cfg.width == 16


def test_signatures_cache_and_file_changes(config):
    oracle = make_main(config, "false", version_base="1.1")
    fast = make_main(config, version_base="1.1")
    for argv in (
        [],
        ["lr=0.1"],
        ["solver=big", "lr=0.1"],
        ["solver=small"],
        ["solver=big", "lr=0.2"],
        ["num_workers=4"],
        ["solver=big", "lr=0.1", "+extra=[1,2]"],
    ):
        reference, actual = oracle.get_xp(argv), fast.get_xp(argv)
        assert actual.sig == reference.sig
        assert actual.delta == reference.delta
        assert OmegaConf.to_container(actual.cfg) == OmegaConf.to_container(reference.cfg)
        assert OmegaConf.to_container(actual.cfg, resolve=True) == OmegaConf.to_container(
            reference.cfg, resolve=True
        )
    base, delta = fast._get_base_config(["solver=big"])
    delta.append(("oops", "value"))
    again, delta = fast._get_base_config(["solver=big", "lr=0.9"])
    assert base is again
    assert delta == [("solver", "big")]
    assert OmegaConf.is_readonly(base)
    xp = fast.get_xp(["solver=big"])
    xp.cfg.optim.momentum = 0
    assert base.optim.momentum == 0.9
    # A changed root is reflected in both the XP and its cached delta baseline.
    path = config / "config.yaml"
    path.write_text(path.read_text().replace("lr: 0.01", "lr: 0.05"))
    refreshed = make_main(config, "false", version_base="1.1")
    assert fast.get_xp(["lr=0.5"]).sig == refreshed.get_xp(["lr=0.5"]).sig


@pytest.mark.parametrize(
    "kwargs, chdir",
    [
        ({}, True),
        ({"version_base": "1.1"}, True),
        ({"version_base": "1.3"}, False),
        ({"version_base": None}, False),
    ],
)
def test_execution_and_saved_config(config, monkeypatch, kwargs, chdir):
    main = make_main(config, **kwargs)
    monkeypatch.setattr(sys, "argv", ["train", "solver=big", "lr=0.3"])
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    previous_handlers = logging.getLogger().handlers[:]
    previous_level = logging.getLogger().level
    xp, cwd, data = main()
    assert cwd == (xp.folder if chdir else config)
    assert data == str(config / "data")
    assert Path.cwd() == config
    assert logging.getLogger().handlers == previous_handlers
    assert logging.getLogger().level == previous_level
    assert "running task" in (xp.folder / "test_fast_parser.log").read_text()
    saved = OmegaConf.load(xp._hydra_config)
    assert saved.lr == 0.3 and saved.alias == 0.3
    assert OmegaConf.to_container(saved)["alias"] == "${lr}"
    assert OmegaConf.load(xp._hydra_config.parent / "overrides.yaml") == xp.argv
    restored = main.get_existing_xp_from_sig(xp.sig)
    assert restored.cfg == xp.cfg and restored.delta == xp.delta


def test_cleanup_on_task_error_and_git_save_paths(config, monkeypatch):
    main = make_main(config)
    monkeypatch.setattr(sys, "argv", ["train"])
    monkeypatch.setenv("_DORA_ORIGINAL_DIR", str(config / "original"))

    def fail(cfg):
        assert to_absolute_path("data") == str(config / "original/data")
        raise RuntimeError("task failed")

    main.main = fail
    before = logging.getLogger().handlers[:]
    with pytest.raises(RuntimeError, match="task failed"):
        main()
    assert Path.cwd() == config
    assert logging.getLogger().handlers == before


def test_nonzero_rank_does_not_replace_metadata(config, monkeypatch):
    main = make_main(config)
    xp = main.get_xp([])
    main.init_xp(xp)
    xp._hydra_config.parent.mkdir()
    xp._hydra_config.write_text("sentinel: true\n")
    monkeypatch.setattr(sys, "argv", ["train"])
    for key, value in {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}.items():
        monkeypatch.setenv(key, value)
    main()
    assert xp._hydra_config.read_text() == "sentinel: true\n"


def test_unsupported_flags_fail_before_creating_an_experiment(config, monkeypatch):
    main = make_main(config)
    monkeypatch.setattr(sys, "argv", ["train", "--multirun"])
    with pytest.raises(UnsupportedFeature, match="use_fast_parser = false"):
        main()
    assert not main.dora.dir.exists()


def test_real_cli_without_hydra(config):
    package = config / "fast_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "train.py").write_text(
        "import importlib.abc, sys\n"
        "class NoHydra(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, *args):\n"
        "        if fullname == 'hydra' or fullname.startswith('hydra.'):\n"
        "            raise AssertionError('Hydra was imported')\n"
        "sys.meta_path.insert(0, NoHydra())\n"
        "from dora import hydra_main, get_xp\n"
        "@hydra_main(config_path='..', config_name='config', version_base='1.1')\n"
        "def main(cfg):\n"
        "    assert cfg.width == 16 and cfg.lr == 0.2\n"
        "    assert 'hydra' not in sys.modules\n"
        "    (get_xp().folder / 'success').write_text(get_xp().sig)\n"
    )
    (config / "dora.toml").write_text(
        '[project]\npackage = "fast_fixture"\nuse_fast_parser = true\n[dora]\ndir = "outputs"\n'
    )
    result = subprocess.run(
        [sys.executable, "-B", "-m", "dora", "run", "solver=big", "lr=0.2"],
        cwd=config,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    folders = list((config / "outputs/xps").iterdir())
    assert len(folders) == 1 and (folders[0] / "success").exists()
    assert json.loads((folders[0] / ".argv.json").read_text()) == ["solver=big", "lr=0.2"]
