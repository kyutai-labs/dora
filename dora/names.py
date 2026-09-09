# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from pathlib import Path
import typing as tp

from .xp import XP


class NamesMixin:
    """Mixin that handles everything related to the naming of experiments.
    """

    def short_name_part(self, key: str, value: tp.Any) -> str:
        """Shorten the name of an XP.
        """
        key_parts = key.split(".")
        short_key_parts = []
        for part in key_parts[:-1]:
            short_key_parts.append(part[:3])
        short_key_parts.append(key_parts[-1])
        key = ".".join(short_key_parts)

        if isinstance(value, Path):
            value = value.name
        if value is True:
            return key
        return f"{key}={value}"

    def get_name_parts(self, xp: XP) -> OrderedDict:
        """Returns name parts, i.e. an OrderedDict from param name -> param value.
        Name parts that don't impact the signature should be ignored.
        """
        raise NotImplementedError()

    def get_name(self, xp: XP) -> str:
        """Returns the XP name.
        """
        if xp.delta is None:
            # Loaded from disk without a persisted delta, so there is nothing to
            # build a name out of. The signature is the one thing always true.
            return xp.sig
        return self.get_names([xp])[-1]

    def _get_short_name(self, parts: OrderedDict, reference: dict = {}):
        out_parts = []
        for key, value in parts.items():
            if key not in reference:
                part = self.short_name_part(key, value)
                out_parts.append(part)
        return " ".join(out_parts)

    def get_names(self, xps: tp.List[XP]) -> tp.Tuple[tp.List[str], str]:
        """Given list of XPs, return individual XP names + base name.
        The common part in all XPs are factored into the base name
        """
        assert len(xps) > 0
        reference = self.get_name_parts(xps[0])
        all_xp_parts = []
        for xp in xps:
            parts = self.get_name_parts(xp)
            for key, val in parts.items():
                if key in reference and reference[key] != val:
                    reference.pop(key)

            missing = set(reference.keys()) - set(parts.keys())
            for key in missing:
                reference.pop(key)
            all_xp_parts.append(parts)

        names = []
        for xp, parts in zip(xps, all_xp_parts):
            if xp.delta is None:
                # An XP loaded from disk that predates the persisted delta: its
                # name is unrecoverable, so fall back to the one thing that is
                # always true about it.
                names.append(xp.sig)
            else:
                names.append(self._get_short_name(parts, reference))

        base_name = self._get_short_name(reference)
        return names, base_name
