# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

from .conf import DoraConfig
from .link import Link
from .utils import jsonable


def _get_sig(delta: list[tp.Any]) -> str:
    # Return signature from a jsonable content.
    sorted_delta = sorted(delta)
    return sha1(json.dumps(sorted_delta).encode("utf8")).hexdigest()[:8]


@dataclass(init=False)
class XP:
    """
    Represent a single experiment, i.e. a specific set of parameters
    that is linked to a unique signature.

    One XP can have multiple runs.
    """

    dora: DoraConfig
    cfg: tp.Any
    argv: list[str]
    sig: str
    delta: list[tuple[str, tp.Any]] | None
    link: Link = field(compare=False)

    def __init__(
        self,
        dora: DoraConfig,
        cfg: tp.Any,
        argv: list[str],
        delta: list[tuple[str, tp.Any]] | None = None,
        sig: str | None = None,
    ):
        self.dora = dora
        self.cfg = cfg
        self.argv = argv
        if delta is not None:
            delta = jsonable([(k, v) for k, v in delta if not dora.is_excluded(k)])
        self.delta = delta
        if sig is None:
            assert delta is not None
            sig = _get_sig(delta)
        self.sig = sig
        self.link = Link(self.folder / self.dora.history)

    @property
    def folder(self) -> Path:
        assert self.sig is not None
        return self.dora.dir / self.dora.xps / self.sig

    @property
    def code_folder(self) -> Path:
        if self.dora.git_save:
            return self.folder / "code"
        else:
            return Path(".")

    @property
    def _xp_submitit(self) -> Path:
        return self.folder / self.dora.shep.submitit_folder

    @property
    def _latest_submitit(self) -> Path:
        return self.folder / self.dora.shep.latest_submitit

    @property
    def submitit(self) -> Path:
        if self._latest_submitit.exists():
            return self._latest_submitit
        else:
            return self._xp_submitit

    @property
    def rendezvous_file(self) -> Path:
        return self.folder / self.dora.rendezvous_file

    @property
    def history(self) -> Path:
        return self.folder / self.dora.history

    @property
    def _argv_cache(self) -> Path:
        return self.folder / ".argv.json"

    @property
    def _delta_cache(self) -> Path:
        """Where the delta is persisted, alongside the argv cache.

        The delta is what the signature is computed from, and it cannot be
        recovered from a stored config alone: it is a diff against a base config
        that is not saved anywhere, and it collapses `+a.b.c=...` additions into
        whole subtrees. Writing it down when the XP is created is the only way to
        recover an experiment's name once its config files have moved on.
        """
        return self.folder / ".delta.json"

    @property
    def _hydra_config(self) -> Path:
        """The composed config Hydra saved for the run that actually happened."""
        return self.folder / ".hydra" / "config.yaml"

    @property
    def _shared_folder(self) -> Path | None:
        if self.dora.shared is not None:
            return self.dora.shared / self.dora.xps / self.sig
        return None

    @property
    def _shared_argv_cache(self) -> Path | None:
        if self._shared_folder is not None:
            return self._shared_folder / ".argv.json"
        return None

    @contextmanager
    def enter(self, stack: bool = False):
        """Context manager, fake being in the XP for its duration.

        Set `stack=True` if you want to allow this to happen from within
        another experiment.

        ..Warning:: For hydra experiment, this will not convert any path
            automatically, or setup loggers etc.
        """
        with _context.enter_xp(self, stack):
            yield


class _Context:
    # Used to keep track of a running XP and be able to provide
    # it on demand with `get_xp`.
    def __init__(self) -> None:
        self._xps: list[XP] = []

    @contextmanager
    def enter_xp(self, xp: XP, stack: bool = False):
        if self._xps and not stack:
            raise RuntimeError("Already in a xp.")
        self._xps.append(xp)
        try:
            yield
        finally:
            self._xps.pop(-1)


_context = _Context()


def get_xp() -> XP:
    """When running from within an XP, returns the XP object.
    Otherwise, raises RuntimeError.
    """
    if not _context._xps:
        raise RuntimeError("Not in a xp!")
    else:
        return _context._xps[-1]


def is_xp() -> bool:
    """Return True if running within an XP."""
    return bool(_context._xps)


def load_xp(dora: "DoraConfig", sig: str) -> XP:
    """Read an experiment straight from its folder.

    The counterpart to `DecoratedMain.get_existing_xp_from_sig` for callers that
    have a `DoraConfig` but do not want to import the project to get one. The
    signature is taken as given -- the folder name is the ground truth -- and
    `cfg` is left as None, since loading it is main-specific.
    """
    xp = XP(dora=dora, cfg=None, argv=[], sig=sig)
    if xp._argv_cache.exists():
        with open(xp._argv_cache) as file:
            xp.argv = json.load(file)
    elif xp._shared_argv_cache is not None and xp._shared_argv_cache.exists():
        with open(xp._shared_argv_cache) as file:
            xp.argv = json.load(file)
    else:
        raise RuntimeError(f"Could not find experiment with signature {sig}")
    if xp._delta_cache.exists():
        with open(xp._delta_cache) as file:
            xp.delta = json.load(file)
    return xp
