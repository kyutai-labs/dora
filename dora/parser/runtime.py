"""Single-run execution for the fast parser; no Hydra runtime or plugins."""

import logging
import re
import sys
from contextlib import contextmanager

from omegaconf import OmegaConf

from ..distrib import get_distrib_spec
from ..git_save import enter_run_dir
from ..main import MainFun
from ..xp import XP
from .errors import UnsupportedFeature


def chdir_for_version(kwargs: dict) -> bool:
    """Preserve Hydra's working-directory default for the decorator's version."""
    extra = kwargs.keys() - {"version_base"}
    if extra:
        raise UnsupportedFeature(f"Unsupported fast-parser decorator arguments: {sorted(extra)}")
    version = kwargs.get("version_base", "1.1")
    if version is None:
        return False
    if not isinstance(version, str) or not re.fullmatch(r"1\.[0-9]+", version):
        raise UnsupportedFeature(f"Unsupported version_base: {version!r}")
    minor = int(version.split(".")[1])
    if minor < 1:
        raise UnsupportedFeature("The fast parser supports Hydra 1.1+ composition semantics")
    return minor == 1


@contextmanager
def _logging(folder, job_name):
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    handlers = []
    try:
        handlers.append(logging.StreamHandler(sys.stdout))
        handlers.append(logging.FileHandler(folder / (job_name + ".log"), encoding="utf8"))
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        for handler in handlers:
            handler.setFormatter(formatter)
        root.handlers = handlers
        root.setLevel(logging.INFO)
        yield
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
        for handler in handlers:
            handler.close()


def run(main: MainFun, xp: XP, job_name: str, *, chdir: bool):
    """Save the unresolved job config, configure logging, and call the task.

    Uses the existing .hydra/config.yaml location so inspection and resuming old
    experiments work with either backend. Only rank zero writes config metadata.
    """
    if get_distrib_spec().rank == 0:
        folder = xp._hydra_config.parent
        folder.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(xp.cfg, xp._hydra_config, resolve=False)
        OmegaConf.save(OmegaConf.create(list(xp.argv)), folder / "overrides.yaml", resolve=False)
    with _logging(xp.folder, job_name), enter_run_dir(xp.folder if chdir else None):
        return main(xp.cfg)
