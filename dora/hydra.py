# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides support for Hydra, in particular the `main` wrapper between
the end user `main` function and Hydra.
"""
from collections import namedtuple, OrderedDict
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import sys
import typing as tp
from unittest import mock

from omegaconf.dictconfig import DictConfig

from .conf import DoraConfig, SlurmConfig, update_from_hydra
from .main import DecoratedMain, MainFun
from .xp import XP, get_xp, is_xp

logger = logging.getLogger(__name__)


def _no_copy(self: tp.Any, memo: tp.Any):
    """Identity stand-in for `DictConfig.__deepcopy__`.

    Used to suppress Hydra's defensive deepcopies in read-only code paths where
    nothing mutates the config.
    """
    return self


_Difference = namedtuple("_Difference", "path key ref other ref_value other_value")


class _NotThere:
    pass


NotThere = _NotThere()


def _compare_config(ref, other, path=[]):
    """
    Given two configs, gives an iterator over all the differences. For each difference,
    this will give a _Difference namedtuple.
    """
    keys = sorted(ref.keys())
    remaining = sorted(set(other.keys()) - set(ref.keys()))
    delta = []
    path.append(None)
    for key in keys:
        path[-1] = key
        ref_value = ref[key]
        assert key in other, f"XP config shouldn't be missing any key. Missing key {key}"
        other_value = other[key]

        if isinstance(ref_value, DictConfig):
            assert isinstance(other_value, DictConfig), \
                "Structure of config should be identical between XPs. "\
                f"Wrong type for {key}, expected DictConfig, got {type(other_value)}."
            yield from _compare_config(ref_value, other_value, path)
        elif other_value != ref_value:
            yield _Difference(list(path), key, ref, other, ref_value, other_value)

    for key in remaining:
        path[-1] = key
        other_value = other[key]
        yield _Difference(list(path), key, ref, other, NotThere, other_value)
    path.pop(-1)
    return delta


def _simplify_argv(argv: tp.Sequence[str]) -> tp.List[str]:
    simplified = []
    seen = set()
    for arg in list(argv)[::-1]:
        assert '=' in arg, f'Argument {arg} does not contain ='
        key, value = arg.split('=', 1)
        key = key.strip()
        if key in seen:
            continue
        else:
            seen.add(key)
            simplified.append(arg)
    return simplified[::-1]


def _dump_key(key):
    if key is None:
        return "null"
    elif isinstance(key, (bool, int, float)):
        return str(key)
    elif isinstance(key, str):
        assert ":" not in key
        return key
    else:
        raise TypeError(f"Unsupported dict key type {type(key)} for key {key}")


def _hydra_value_as_override(value):
    # hydra doesn't support parsing dict with the json format, so for now
    # we have to use a custom function to dump a value.
    if value is None:
        return "null"
    elif isinstance(value, (bool, int, float, str)):
        return json.dumps(value)
    elif isinstance(value, dict):
        return "{" + ", ".join(
            f"{_dump_key(key)}: {_hydra_value_as_override(val)}"
            for key, val in value.items()
        ) + "}"
    elif isinstance(value, (list, tuple)):
        return "[" + ", ".join(_hydra_value_as_override(val) for val in value) + "]"
    else:
        raise TypeError(f"Unsupported value type {type(value)} for value {value}")


class HydraMain(DecoratedMain):
    _slow = True
    use_fast_parser = False

    def __init__(self, main: MainFun, config_name: str, config_path: str, **kwargs):
        self.config_name = config_name
        self.config_path = config_path
        self.hydra_kwargs = kwargs

        module = main.__module__
        if module == "__main__":
            spec = sys.modules[module].__spec__
            if spec is None:
                module_path = sys.argv[0]
                self._job_name = module_path.rsplit(".", 2)[1]
            else:
                assert spec.origin is not None
                module_path = spec.origin
                module = spec.name
                self._job_name = module.rsplit(".", 1)[1]
        else:
            spec = find_spec(module)
            assert spec is not None and spec.origin is not None
            module_path = spec.origin
            self._job_name = module.rsplit(".", 1)[1]
        self.full_config_path = Path(module_path).parent.resolve()
        if config_path is not None:
            self.full_config_path = self.full_config_path / config_path

        from . import project
        conf = project.load()
        self.use_fast_parser = conf.use_fast_parser if conf is not None else False
        if self.use_fast_parser:
            from .parser import ConfigParser
            from .parser.runtime import chdir_for_version
            self._fast_chdir = chdir_for_version(kwargs)
            self._parser = ConfigParser(self.full_config_path, self.config_name)

        self._initialized = False
        self._base_cfg = self._get_config()
        self._config_groups = self._get_config_groups()
        dora = self._get_dora()
        super().__init__(main, dora)
        # this is a really dirty hack to make Hydra believe that this is
        # coming from the __main__ module, as it would usually be.
        # This allows to use relative paths for config_path.
        main.__module__ = "__main__"

    def _get_dora(self) -> DoraConfig:
        """Dora's own settings: from `dora.toml` if there is one, then from the
        `dora:` block of the composed config.

        The YAML block still wins where both say something, so nothing changes
        for a project without a `dora.toml`. With one, the block can go away
        entirely -- which is the point, since restating the settings in two
        places is how they drift.
        """
        from . import project
        conf = project.load()
        dora = None
        if conf is not None:
            # require_dir=False: `dir` may legitimately come from the YAML below
            # or be set by the project after import, but `exclude` decides
            # signatures and must not be silently dropped.
            dora = conf.dora_config(require_dir=False)
        if dora is None:
            dora = DoraConfig()
        if hasattr(self._base_cfg, "dora"):
            update_from_hydra(dora, self._base_cfg.dora)
        dora.exclude += ["dora.*", "slurm.*"]
        dora.dir = Path(dora.dir)
        return dora

    def get_slurm_config(self) -> SlurmConfig:
        """Return default Slurm config for the launch and grid actions.
        """
        slurm = SlurmConfig()
        if hasattr(self._base_cfg, "slurm"):
            update_from_hydra(slurm, self._base_cfg.slurm)
        return slurm

    def get_xp(self, argv: tp.Sequence[str]):
        argv = _simplify_argv(argv)
        cfg = self._get_config(argv)
        base, delta = self._get_base_config(argv)
        delta += self._get_delta(base, cfg)
        xp = XP(dora=self.dora, cfg=cfg, argv=argv, delta=delta)
        return xp

    def value_to_argv(self, arg: tp.Any) -> tp.List[str]:
        # Here we get the raw stuff from what is passed to the grid launcher.
        # arg is either a str (in which case it is a raw override)
        # or a dict, in which case each entry is an override,
        # or a list of dict or a list of str.
        argv = []
        if isinstance(arg, str):
            argv.append(arg)
        elif isinstance(arg, dict):
            for key, value in arg.items():
                if key not in self._config_groups:
                    # We need to convert the value using a custom function
                    # to respect how Hydra parses overrides.
                    value = _hydra_value_as_override(value)
                argv.append(f"{key}={value}")
        elif isinstance(arg, (list, tuple)):
            for part in arg:
                argv += self.value_to_argv(part)
        else:
            raise ValueError(f"Can only process dict, tuple, lists and str, but got {arg}")
        return argv

    def _load_existing_cfg(self, xp: XP) -> tp.Any:
        """Load the config Hydra saved for the run, skipping composition entirely.

        Hydra writes the fully composed config to `<xp>/.hydra/config.yaml`
        before the job starts. Reading it back is both faster than recomposing
        (about 15ms against 210ms, since composition re-reads and re-wraps the
        same YAML files on every call) and more truthful, because it is the
        config the job actually ran with rather than what today's config tree
        would produce from the same arguments.

        The file is stored unresolved, so `${...}` interpolations survive and
        `OmegaConf.create` is needed to make them resolve on access.
        """
        if not xp._hydra_config.exists():
            return None
        import yaml
        from omegaconf import OmegaConf
        try:
            # CSafeLoader where libyaml is available; OmegaConf.load would use
            # the pure-Python loader, which is ~9x slower on these files.
            from yaml import CSafeLoader as SafeLoader  # type: ignore
        except ImportError:
            from yaml import SafeLoader  # type: ignore
        with open(xp._hydra_config) as fileobj:
            raw = yaml.load(fileobj, Loader=SafeLoader)
        return OmegaConf.create(raw)

    def get_name_parts(self, xp: XP) -> OrderedDict:
        parts: OrderedDict = OrderedDict()
        if xp.delta is None:
            # Loaded from disk without a persisted delta; `get_names` falls back
            # to the signature for these.
            return parts
        for name, value in xp.delta:
            parts[name] = value
        return parts

    def _main(self):
        if self.use_fast_parser:
            from .parser.runtime import run
            return run(self.main, get_xp(), self._job_name, chdir=self._fast_chdir)

        import hydra
        from hydra.core.global_hydra import GlobalHydra
        # Imported here rather than at module scope: `dora.distrib` pulls in torch
        # (~1.5s), and this is the only place in `dora.hydra` that needs it. Keeping
        # it out of the import graph is what makes `import dora` cheap.
        from .distrib import get_distrib_spec
        if is_xp():
            run_dir = f"hydra.run.dir={get_xp().folder}"
            sys.argv.append(run_dir)
            if get_distrib_spec().rank > 0:
                sys.argv.append("hydra.output_subdir=null")
        try:
            return hydra.main(
                config_name=self.config_name,
                config_path=self.config_path,
                **self.hydra_kwargs)(self.main)()
        finally:
            if is_xp():
                sys.argv.remove(run_dir)
            # `hydra.main` leaves GlobalHydra initialized on the way out, which
            # makes any later `initialize_config_dir` in the same process raise.
            # That bites anything running more than one XP per process -- test
            # suites, notebooks -- and consumers have had to clear it themselves.
            GlobalHydra.instance().clear()

    def _get_config_groups(self, fast: bool = True) -> tp.List[str]:
        """List the Hydra config groups, used to tell `group=value` overrides
        from plain `dotted.key=value` ones.

        `fast` suppresses Hydra's internal deepcopies; pass False to get the
        unaccelerated answer, which the tests compare against.
        """
        if self.use_fast_parser:
            return sorted(path.relative_to(self.full_config_path).as_posix()
                          for path in self.full_config_path.rglob("*") if path.is_dir())
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            gh = GlobalHydra.instance().hydra
            assert gh is not None
            if not fast:
                return list(gh.list_all_config_groups())
            # `list_all_config_groups` builds a fresh CachingConfigRepository per
            # group, and each one deepcopies the whole config-source list -- about
            # two thirds of the cost of this call, which runs at import time for
            # every Dora invocation. Nothing here mutates a config, we only read
            # group names, so making the copy a no-op is safe.
            # `test_hydra.py::test_config_groups_unaffected_by_no_copy` pins that.
            with mock.patch.object(DictConfig, "__deepcopy__", _no_copy):
                return list(gh.list_all_config_groups())

    def _is_active(self, argv: tp.List[str]) -> bool:
        if self.use_fast_parser:
            from .parser import UnsupportedFeature
            for arg in argv:
                if arg.startswith("-"):
                    raise UnsupportedFeature(
                        f"Hydra CLI option {arg!r} requires use_fast_parser = false; "
                        "the fast parser accepts config overrides only")
            return True
        if '-m' in argv or '--multirun' in argv:
            return False
        return True

    def _get_base_config(
            self, overrides: tp.List[str] = []
            ) -> tp.Tuple[DictConfig, tp.List[tp.Tuple[str, str]]]:
        """
        Return base config based on composition, along with delta for the
        composition overrides.
        """
        if self.use_fast_parser:
            # Keep Dora's existing group delta semantics, including argument order.
            to_keep = []
            delta: tp.List[tp.Tuple[str, str]] = []
            for arg in overrides:
                group, _, value = arg.partition("=")
                if group in self._config_groups:
                    to_keep.append(arg)
                    delta = [(g, v) for g, v in delta if g != group]
                    delta.append((group, value))
            return self._parser.compose_base_config(to_keep), delta

        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            gh = GlobalHydra.instance().hydra
            assert gh is not None
            to_keep = []
            delta = []
            for arg in overrides:
                for group in self._config_groups:
                    if arg.startswith(f'{group}='):
                        to_keep.append(arg)
                        _, value = arg.split('=', 1)
                        delta = [(g, v) for g, v in delta if g != group]
                        delta.append((group, value))
            if not to_keep:
                return self._base_cfg, []
            cfg = self._get_config_noinit(to_keep)
            return cfg, delta

    def _get_config(self,
                    overrides: tp.List[str] = []) -> DictConfig:
        """
        Internal method, returns the config for the given override,
        but without the dora.sig field filled.
        """
        if self.use_fast_parser:
            return self._parser.compose_config(overrides)
        from hydra import initialize_config_dir
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            return self._get_config_noinit(overrides)

    def _get_config_noinit(self, overrides: tp.List[str] = []) -> DictConfig:
        from hydra import compose
        return compose(self.config_name, overrides)  # type: ignore

    def _get_delta(self, init: DictConfig, other: DictConfig):
        """
        Returns an iterator over all the differences between the init and other config.
        """
        delta = []
        for diff in _compare_config(init, other):
            name = ".".join(diff.path)
            delta.append((name, diff.other_value))
        return delta


def hydra_main(config_name: str, config_path: str, **kwargs):
    """Wrap your main function with this.
    You can pass extra kwargs, e.g. `version_base` introduced in 1.2.
    Set [project] use_fast_parser = true in dora.toml to use dora.parser
    for composition and execution without Hydra. The default is false.
    """
    def _decorator(main: MainFun):
        return HydraMain(main, config_name=config_name, config_path=config_path,
                         **kwargs)
    return _decorator
