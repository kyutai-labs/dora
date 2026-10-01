# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Signature stability tests.

`test_golden_corpus` is the real net: point `DORA_GOLDEN_CORPUS` at a corpus
built from a project's own experiments (see `dora.tests.golden`) and every
signature in it is recomputed. It is skipped when unset, so CI stays fast and
contributors without a corpus are not blocked.

The always-on tests below pin the pieces of signature computation that do not
need a project: the hash itself, and the exclusion filtering applied before it.
"""

import os
from pathlib import Path

import pytest

from ..conf import DoraConfig
from ..xp import XP, _get_sig
from . import golden


def test_sig_is_order_independent():
    # The delta is sorted before hashing, so argv order must not matter.
    assert _get_sig([("b", 2), ("a", 1)]) == _get_sig([("a", 1), ("b", 2)])


def test_sig_is_stable():
    # A literal, so a change to the hashing scheme cannot slip through silently.
    assert _get_sig([("optim.lr", 0.0001), ("solver", "lm/default")]) == "feaac075"


def test_excluded_keys_do_not_change_the_sig():
    dora = DoraConfig(exclude=["device", "wandb.*"])
    delta = [("optim.lr", 1e-4)]
    noisy = delta + [("device", "cuda"), ("wandb.project", "x")]
    assert (
        XP(dora=dora, cfg=None, argv=[], delta=delta).sig
        == XP(dora=dora, cfg=None, argv=[], delta=noisy).sig
    )


@pytest.mark.skipif(
    not os.environ.get("DORA_GOLDEN_CORPUS"),
    reason="set DORA_GOLDEN_CORPUS to a corpus built by dora.tests.golden",
)
def test_golden_corpus():
    from .._utils import get_main

    path = Path(os.environ["DORA_GOLDEN_CORPUS"])
    corpus = golden.load(path)
    assert corpus, f"empty corpus at {path}"
    report = golden.check(get_main(), corpus)
    detail = "\n".join(f"  {sig}: {why}" for sig, why in report.drifted + report.broke)
    assert not report.failed, f"{report.summary()}\n{detail}"
