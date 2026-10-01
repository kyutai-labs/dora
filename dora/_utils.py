# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Internal utilities, likely shouldn't be called from outside."""

import os
import sys
from pathlib import Path

from . import project
from .conf import DoraConfig
from .log import fatal
from .main import DecoratedMain
from .utils import import_or_fatal


def _find_package(main_module: str, cwd: Path | None = None):
    cwd = cwd or Path(".")
    candidates = []
    for child in cwd.iterdir():
        if (
            child.is_dir()
            and (child / "__init__.py").exists()
            and (child / f"{main_module}.py").exists()
        ):
            candidates.append(child.name)
    if len(candidates) == 0:
        fatal(
            "Could not find a training package. Use -P, or set DORA_PACKAGE to set the "
            "package. Use --main_module or set DORA_MAIN_MODULE to set the module to "
            "be excecuted inside the defined package."
        )
    elif len(candidates) == 1:
        return candidates[0]
    else:
        fatal(
            f"Found multiple candidates: {', '.join(candidates)}. "
            "Use -P, or set DORA_PACKAGE to set package being searched. "
            "Use --main_module or set DORA_MAIN_MODULE to set the module being searched "
            "inside the package."
        )


def get_main(main_module: str | None = None, package: str | None = None):
    """Import the project's training module and return its `DecoratedMain`.

    Precedence for locating it: explicit argument, then environment, then
    `dora.toml`, then scanning for a package that contains the module.
    """
    conf = project.load()
    root = conf.root if conf is not None else Path.cwd()

    if main_module is None:
        main_module = (
            os.environ.get("DORA_MAIN_MODULE") or (conf.main_module if conf else None) or "train"
        )
    if package is None:
        package = os.environ.get("DORA_PACKAGE")
        if package is None and conf is not None:
            package = conf.package
        if package is None:
            package = _find_package(main_module, root)
    module_name = package + "." + main_module
    # The repository root, not the working directory: with a `dora.toml` to
    # locate it, `dora` works from anywhere inside the project.
    sys.path.insert(0, str(root))
    module = import_or_fatal(module_name)
    try:
        main = module.main
    except AttributeError:
        fatal(f"Could not find function `main` in {module_name}.")

    if not isinstance(main, DecoratedMain):
        fatal(f"{module_name}.main was not decorated with `dora.main`.")
    return main


def get_dora_config() -> DoraConfig | None:
    """The project's `DoraConfig` without importing the project, if possible.

    Read-only commands need little more than the experiment directory, and
    importing a research codebase to learn it costs seconds. Returns None when
    `dora.toml` is absent or cannot resolve the directory, and the caller should
    then fall back to `get_main().dora`.
    """
    conf = project.load()
    if conf is None:
        return None
    return conf.dora_config()
