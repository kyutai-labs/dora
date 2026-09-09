# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Static project settings, read from a `dora.toml` at the repository root.

Dora learns where a project keeps its experiments by importing the project's
training module and inspecting the `DecoratedMain` it defines. For anything that
only wants to *read* -- what is the state of this grid, why did this job die,
what were the last metrics -- that is a wildly disproportionate price: importing
a research codebase pulls in torch and the rest, which on a real project is ten
seconds and change, to answer a question that is a few file reads.

Almost everything needed is static per project: which package holds the training
module, where experiments live, which parameters are excluded from signatures.
`dora.toml` states it once, so read-only commands can skip the import entirely::

    [project]
    package     = "audiocraft"
    main_module = "train"
    config_path = "config"
    config_name = "config"

    [dora]
    dir     = "${env:AUDIOCRAFT_DORA_DIR}"
    exclude = ["device", "wandb.*"]
    git_save = true

Values may interpolate environment variables as `${env:VAR}`, optionally with a
fallback: `${env:VAR:-/tmp/default}`. A `dir` that cannot be resolved is left
unset rather than guessed at, and callers fall back to importing the project.

The file is optional. Without it everything behaves as before, just slower.
"""
from dataclasses import fields
import os
from pathlib import Path
import re
import typing as tp

from .conf import DoraConfig

TOML_NAME = "dora.toml"

# ${env:VAR} or ${env:VAR:-fallback}
_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ProjectConfigError(RuntimeError):
    """Raised when a `dora.toml` exists but cannot be used."""


class _Unresolved(str):
    """A value with an unresolvable `${env:...}` left in it."""


def _interpolate(value: tp.Any) -> tp.Any:
    """Substitute `${env:VAR}` references, recursively through lists and dicts."""
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if not isinstance(value, str):
        return value

    missing = False

    def _sub(match: "re.Match") -> str:
        nonlocal missing
        name, fallback = match.group(1), match.group(2)
        env = os.environ.get(name)
        if env is not None:
            return env
        if fallback is not None:
            return fallback
        missing = True
        return ""

    out = _ENV_RE.sub(_sub, value)
    return _Unresolved(out) if missing else out


def find_toml(start: tp.Optional[Path] = None) -> tp.Optional[Path]:
    """Look for a `dora.toml`, walking up from `start` (default: cwd).

    Walking up is deliberate: `dora` should work from a subdirectory of the
    repository, which the package-scanning fallback does not manage.
    """
    current = (start or Path(".")).resolve()
    for folder in [current, *current.parents]:
        candidate = folder / TOML_NAME
        if candidate.is_file():
            return candidate
    return None


class ProjectConfig:
    """Everything `dora.toml` says about a project."""

    def __init__(self, path: Path, raw: dict):
        self.path = path
        self.root = path.parent
        project = raw.get("project", {})
        self.package: tp.Optional[str] = project.get("package")
        self.main_module: tp.Optional[str] = project.get("main_module")
        self.config_path: tp.Optional[str] = project.get("config_path")
        self.config_name: tp.Optional[str] = project.get("config_name")
        self.hydra_kwargs: dict = project.get("hydra", {})
        self._dora: dict = raw.get("dora", {})

    def dora_config(self) -> tp.Optional[DoraConfig]:
        """Build a `DoraConfig`, or None if `dir` could not be resolved.

        Returning None rather than a half-built config is the point: a wrong
        experiment directory would silently look at nothing, which is worse than
        being slow.
        """
        if not self._dora:
            return None
        known = {f.name for f in fields(DoraConfig)} | {"dir_probe"}
        unknown = set(self._dora) - known
        if unknown:
            raise ProjectConfigError(
                f"{self.path}: unknown [dora] keys: {', '.join(sorted(unknown))}")

        probes = self._dora.get("dir_probe", [])
        values = {key: _interpolate(value)
                  for key, value in self._dora.items() if key != "dir_probe"}
        directory = values.get("dir")
        if directory is None or isinstance(directory, _Unresolved) or not str(directory):
            # Fall back to probing: the first entry whose `probe` path exists
            # wins. Projects that run on several clusters pick their experiment
            # directory by looking for a marker path, and this lets them say so
            # statically instead of forcing an import to find out.
            directory = None
            for entry in probes:
                try:
                    marker, candidate = entry["probe"], entry["dir"]
                except (KeyError, TypeError):
                    raise ProjectConfigError(
                        f"{self.path}: each [[dora.dir_probe]] needs `probe` and `dir`.")
                if Path(_interpolate(marker)).exists():
                    candidate = _interpolate(candidate)
                    if not isinstance(candidate, _Unresolved) and str(candidate):
                        directory = candidate
                    break
            if directory is None:
                return None
        # A relative dir is relative to the repository root, not to wherever the
        # command happened to be run from.
        values["dir"] = Path(directory)
        if not values["dir"].is_absolute():
            values["dir"] = (self.root / values["dir"]).resolve()
        for key in ("shared",):
            if values.get(key) is not None and not isinstance(values[key], _Unresolved):
                values[key] = Path(values[key])
            else:
                values.pop(key, None)
        return DoraConfig(**values)


def load(start: tp.Optional[Path] = None) -> tp.Optional[ProjectConfig]:
    """Find and parse the nearest `dora.toml`, or None if there is none."""
    path = find_toml(start)
    if path is None:
        return None
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            raise ProjectConfigError(
                f"Found {path} but no TOML parser; install `tomli` or use Python 3.11+.")
    with open(path, "rb") as fileobj:
        raw = tomllib.load(fileobj)
    return ProjectConfig(path, raw)
