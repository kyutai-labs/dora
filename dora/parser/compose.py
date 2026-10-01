"""Filesystem-only Hydra subset, composed as ordinary Python containers."""

from __future__ import annotations

import posixpath
import re
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig

from .errors import ConfigError, UnsupportedFeature
from .overrides import Override, parse_override
from .yaml_loader import load_yaml

_ABSENT = object()
_HEADER = re.compile(r"^\s*#\s*@package\s+(\S+)\s*$", re.MULTILINE)


def _merge(target: dict, source: dict, *, strict=False):
    for key, value in source.items():
        previous = target.get(key, _ABSENT)
        if strict and previous is _ABSENT:
            raise ConfigError(f"Unknown key {key!r}; use '+' to add it")
        if isinstance(value, str) and value == "???" and previous is not _ABSENT:
            continue
        if _interpolation(previous) and isinstance(value, (dict, list)):
            raise UnsupportedFeature(f"Merging a container into an interpolation at {key!r}")
        if isinstance(previous, dict) and isinstance(value, dict):
            _merge(previous, value, strict=strict)
        elif (
            isinstance(previous, (dict, list))
            and isinstance(value, (dict, list))
            and type(previous) is not type(value)
        ):
            raise ConfigError(
                f"Cannot merge {type(value).__name__} into {type(previous).__name__} at {key!r}"
            )
        else:
            target[key] = deepcopy(value)


def _interpolation(value):
    return isinstance(value, str) and "${" in value


def _apply(config: dict, override: Override):
    if "@" in override.key or "/" in override.key:
        raise ConfigError(f"Unknown config group: {override.key!r}")
    parts = override.key.split(".")
    parent: Any = config
    force = override.operation in ("add", "force")
    for part in parts[:-1]:
        if _interpolation(parent):
            raise UnsupportedFeature(f"Updating through an interpolation: {override.key}")
        if isinstance(parent, list):
            try:
                parent = parent[int(part)]
            except (ValueError, IndexError) as exc:
                raise ConfigError(f"Invalid list index in {override.key!r}") from exc
        elif isinstance(parent, dict):
            value = parent.get(part, _ABSENT)
            if value is _ABSENT or value is None or not isinstance(value, (dict, list)):
                if _interpolation(value):
                    raise UnsupportedFeature(f"Updating through an interpolation: {override.key}")
                if not force:
                    raise ConfigError(
                        f"Unknown or non-container path {override.key!r}; use '+' to add it"
                    )
                parent[part] = {}
            parent = parent[part]
        else:
            raise ConfigError(f"Non-container path in {override.key!r}")
    key: Any = parts[-1]
    if isinstance(parent, list):
        try:
            key = int(key)
            if key < 0:
                raise ValueError
            old = parent[key]
        except (ValueError, IndexError) as exc:
            raise ConfigError(f"Invalid list index in {override.key!r}") from exc
    elif isinstance(parent, dict):
        old = parent.get(key, _ABSENT)
    else:
        if _interpolation(parent):
            raise UnsupportedFeature(f"Updating through an interpolation: {override.key}")
        raise ConfigError(f"Non-container path in {override.key!r}")

    value = override.value
    if override.operation == "delete":
        if _interpolation(old):
            raise UnsupportedFeature(f"Deleting an interpolated value: {override.key}")
        if old is _ABSENT or old is None or old == "???":
            raise ConfigError(f"Cannot delete absent/null key {override.key!r}")
        if value is not None and old != value:
            raise ConfigError(f"Delete value does not match {override.key!r}")
        del parent[key]
        return
    if override.operation == "add" and not isinstance(value, (dict, list)):
        if _interpolation(old):
            raise UnsupportedFeature(f"Adding over an interpolated value: {override.key}")
        if old is not _ABSENT and old is not None and old != "???":
            raise ConfigError(f"Key {override.key!r} already exists; use '++' to replace it")
    if old is _ABSENT and not force:
        raise ConfigError(f"Unknown key {override.key!r}; use '+' to add it")
    if _interpolation(old) and isinstance(value, (dict, list)):
        raise UnsupportedFeature(f"Merging a container into an interpolation: {override.key}")
    if isinstance(old, dict) and isinstance(value, dict):
        _merge(old, value, strict=not force)
    else:
        parent[key] = deepcopy(value)


def _join(parent, child):
    return child.lstrip("/") if child.startswith("/") else "/".join(filter(None, (parent, child)))


def _package(parent, value):
    if value == "_here_":
        return parent
    result = ".".join(filter(None, (parent, value)))
    # Hydra also accepts Audium's historical '__global__' spelling.
    if "_global_" in result:
        result = result[result.rfind("_global_") + len("_global_") + 1 :]
    return result


@dataclass
class _Document:
    data: dict
    defaults: list
    header: str | None


@dataclass
class _Base:
    data: dict
    dependencies: dict
    config: Any = None


class ConfigParser:
    """Reusable composer for a directory of YAML configs.

    ``compose`` returns an independent dict with unresolved interpolations.
    ``compose_config`` creates an OmegaConf DictConfig only at the API boundary.
    Files and group compositions are cached per instance. Cached dependencies
    are stat-checked on each call by default; ``clear_cache`` starts fresh.
    Instances are intended for one thread each.
    """

    def __init__(self, config_dir, config_name="config", *, cache_size=32, check_files=True):
        self.config_dir = Path(config_dir).expanduser().resolve()
        if not self.config_dir.is_dir():
            raise ConfigError(f"Config directory does not exist: {self.config_dir}")
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.config_name = config_name
        self.cache_size = cache_size
        self.check_files = check_files
        self._files: dict[Path, tuple[tuple[int, int, int], _Document]] = {}
        self._bases: OrderedDict[tuple[str, tuple[str, ...]], _Base] = OrderedDict()

    def clear_cache(self):
        self._files.clear()
        self._bases.clear()

    def is_group(self, key):
        name = key.split("@", 1)[0]
        if name == "hydra" or name.startswith(("hydra/", "hydra.")):
            raise UnsupportedFeature("Hydra runtime configuration and plugins are not supported")
        return self._path(name).is_dir()

    def group_overrides(self, overrides: Iterable[str]):
        """Return group-selection arguments (useful for Dora's base-config delta)."""
        result = []
        for arg in overrides:
            override = parse_override(arg)
            if self.is_group(override.key) and not isinstance(override.value, dict):
                result.append(arg)
        return result

    def _path(self, name):
        if not isinstance(name, str) or not name or "${" in name:
            raise UnsupportedFeature(f"Unsupported config path: {name!r}")
        # Config paths are logical names, never arbitrary filesystem paths.
        normalized = posixpath.normpath(name)
        if normalized == ".." or normalized.startswith("../") or name.startswith("/"):
            raise ConfigError(f"Config path must stay inside the config directory: {name!r}")
        return self.config_dir / normalized

    @staticmethod
    def _stamp(path):
        try:
            info = path.stat()
            return (info.st_mtime_ns, info.st_size, info.st_ino)
        except FileNotFoundError:
            return None

    def _load(self, name, dependencies, optional=False):
        if not name.endswith((".yaml", ".yml")):
            name += ".yaml"
        path = self._path(name)
        stamp = self._stamp(path)
        dependencies[path] = stamp
        if stamp is None:
            if optional:
                return None
            raise ConfigError(f"Config not found: {path}")
        cached = self._files.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            text = path.read_text(encoding="utf8")
            raw = load_yaml(text)
        except Exception as exc:
            raise ConfigError(f"Cannot load {path}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"Config must be a mapping: {path}")
        if "hydra" in raw:
            raise UnsupportedFeature(f"Hydra runtime/search-path settings in {path}")
        defaults = raw.pop("defaults", [])
        if not isinstance(defaults, list):
            raise ConfigError(f"defaults must be a list: {path}")
        header = _HEADER.search(text)
        doc = _Document(raw, defaults, header[1] if header else None)
        self._files[path] = stamp, doc
        return doc

    def _base(self, groups):
        cache_key = self.config_name, tuple(groups)
        cached = self._bases.get(cache_key)
        if cached and (
            not self.check_files
            or all(self._stamp(p) == stamp for p, stamp in cached.dependencies.items())
        ):
            self._bases.move_to_end(cache_key)
            return cached
        overrides = [parse_override(arg) for arg in groups]
        selections = {}
        deletes = {}
        appended = []
        required = set()
        for override in overrides:
            key = override.key
            if override.operation != "delete" and not isinstance(override.value, (str, list)):
                raise ConfigError(f"CLI config group {key} requires a string choice")
            if override.operation == "force":
                raise UnsupportedFeature("'++' is not supported for config groups")
            if override.operation == "add":
                appended.append({key: override.value})
            elif override.operation == "delete":
                deletes[key] = override.value
                required.add(key)
            else:
                selections[key] = override.value
                required.add(key)
        dependencies = {}
        seen = {}
        deleted = set()
        stack = []

        def group_info(key, directory, package):
            ref, sep, relocation = key.partition("@")
            group = _join(directory, ref)
            default_package = _package(
                package, relocation if sep else ref.lstrip("/").replace("/", ".")
            )
            identity = group
            if default_package != group.replace("/", "."):
                identity += "@" + (default_package or "_global_")
            return group, relocation if sep else None, default_package, identity

        def expand(
            name,
            parent_package="",
            default_package="",
            relocation=None,
            optional=False,
            group_directory=None,
        ):
            if name in stack:
                raise ConfigError("Defaults cycle: " + " -> ".join(stack + [name]))
            if len(stack) >= 100:
                raise ConfigError("Defaults nesting exceeds 100 configs")
            doc = self._load(name, dependencies, optional)
            if doc is None:
                return []
            package = default_package
            if relocation is not None:
                package = _package(parent_package, relocation)
            elif doc.header is not None:
                if "_group_" in doc.header or "_name_" in doc.header:
                    raise UnsupportedFeature(f"Legacy package substitution in {name}")
                package = _package("", doc.header)
            directory = name.rpartition("/")[0] if group_directory is None else group_directory
            defaults = list(doc.defaults)
            if defaults.count("_self_") > 1:
                raise ConfigError(f"Duplicate _self_ in {name}")
            if "_self_" not in defaults:
                defaults.append("_self_")
            if not stack:
                defaults.extend(appended)
            entries = []
            found_override = False
            for entry in defaults:
                if isinstance(entry, str):
                    if found_override and entry != "_self_":
                        raise ConfigError(f"Defaults overrides must come last in {name}")
                    entries.append(entry)
                    continue
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ConfigError(f"Invalid defaults entry in {name}: {entry!r}")
                key, value = next(iter(entry.items()))
                if not isinstance(key, str):
                    raise ConfigError(f"Defaults keys must be strings in {name}")
                modifier = ""
                if key.startswith(("override ", "optional ")):
                    modifier, key = key.split(" ", 1)
                info = group_info(key, directory, package)
                if modifier == "override":
                    found_override = True
                    selections.setdefault(info[3], value)
                    required.add(info[3])
                else:
                    if found_override and entry not in appended:
                        raise ConfigError(f"Defaults overrides must come last in {name}")
                    entries.append((info, value, modifier == "optional"))
            # Resolve later subtrees first: their overrides can select earlier
            # siblings (e.g. solver overriding the root's dset selection).
            stack.append(name)
            blocks = []
            try:
                for entry in reversed(entries):
                    if entry == "_self_":
                        blocks.append([(package, doc.data)])
                    elif isinstance(entry, str):
                        ref, sep, reloc = entry.partition("@")
                        path = _join(directory, ref)
                        default = _package(
                            package, ref.lstrip("/").rpartition("/")[0].replace("/", ".")
                        )
                        blocks.append(expand(path, package, default, reloc if sep else None))
                    else:
                        (group, reloc, default, identity), choice, optional_entry = entry
                        if identity in deletes and (
                            deletes[identity] is None or deletes[identity] == choice
                        ):
                            deleted.add(identity)
                            continue
                        choice = selections.get(identity, choice)
                        if identity in seen:
                            raise ConfigError(f"Config group selected more than once: {identity}")
                        seen[identity] = choice
                        if choice is None:
                            continue
                        if isinstance(choice, list):
                            raise UnsupportedFeature(
                                "Multiple choices in a defaults group are not supported"
                            )
                        if not isinstance(choice, str):
                            raise ConfigError(
                                f"Config group {identity} needs a string or null choice"
                            )
                        if choice == "???":
                            raise ConfigError(f"Config group {identity} requires a choice")
                        if "${" in choice:
                            raise UnsupportedFeature(
                                "Interpolated defaults choices are not supported"
                            )
                        blocks.append(
                            expand(
                                group + "/" + choice, package, default, reloc, optional_entry, group
                            )
                        )
            finally:
                stack.pop()
            return [item for block in reversed(blocks) for item in block]

        plan = expand(self.config_name)
        unused = required - seen.keys() - deleted
        if unused:
            raise ConfigError(
                f"Overrides did not match the defaults list: {sorted(unused)}; "
                "use '+' to append a group"
            )
        unmatched_deletes = deletes.keys() - deleted
        if unmatched_deletes:
            raise ConfigError(f"Group deletion did not match: {sorted(unmatched_deletes)}")
        data = {}
        for package, content in plan:
            wrapped = content
            if package:
                for part in reversed(package.split(".")):
                    wrapped = {part: wrapped}
            _merge(data, wrapped)
        entry = _Base(data, dependencies)
        if self.cache_size:
            self._bases[cache_key] = entry
            self._bases.move_to_end(cache_key)
            while len(self._bases) > self.cache_size:
                self._bases.popitem(last=False)
        return entry

    def compose(self, overrides: Iterable[str] = ()) -> dict:
        """Compose without resolving ``${...}`` or wrapping containers."""
        groups, values = [], []
        for arg in overrides:
            override = parse_override(arg)
            if self.is_group(override.key) and not isinstance(override.value, dict):
                groups.append(arg)
            else:
                values.append(override)
        result = deepcopy(self._base(groups).data)
        for override in values:
            _apply(result, override)
        return result

    def compose_base_config(self, overrides: Iterable[str] = ()) -> DictConfig:
        """Return a shared, read-only DictConfig for the selected config groups.

        Ordinary value overrides are ignored. Reuses the same bounded cache and
        dependency checks as compose(), including absent optional configs.
        Interpolations stay lazy; resolver caches are cleared on each call.
        Treat this as a borrowed reference: do not disable its readonly flag.
        Use compose_config() for an independent mutable experiment config.
        """
        from omegaconf import OmegaConf

        entry = self._base(self.group_overrides(overrides))
        if entry.config is None:
            config = OmegaConf.create(entry.data)
            OmegaConf.set_struct(config, True)
            OmegaConf.set_readonly(config, True)
            entry.config = config
        else:
            # Match freshly constructed bases for resolvers with use_cache=True.
            OmegaConf.clear_cache(entry.config)
        return entry.config

    def compose_config(self, overrides: Iterable[str] = ()) -> DictConfig:
        """Compose an OmegaConf DictConfig with lazy interpolation and struct mode."""
        from omegaconf import DictConfig, OmegaConf

        config = OmegaConf.create(self.compose(overrides))
        assert isinstance(config, DictConfig)
        OmegaConf.set_struct(config, True)
        return config


def compose(config_dir, overrides=(), *, config_name="config") -> dict:
    """One-shot convenience API; reuse ConfigParser for grids."""
    return ConfigParser(config_dir, config_name).compose(overrides)
