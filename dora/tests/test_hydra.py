# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from ..git_save import assign_clone, enter_clone, get_new_clone, to_absolute_path
from ..hydra import hydra_main
from ..xp import XP, get_xp

_ret = None

current_path = Path.cwd()


def _main(cfg):
    global _ret
    xp = get_xp()
    xp.link.push_metrics({"loss": 0.1})
    _ret = xp  # hydra does not support return values
    assert to_absolute_path(".") == str(current_path), (to_absolute_path("."), current_path)


def get_main(tmpdir):
    tmpdir = Path(str(tmpdir))
    dora_main = hydra_main(config_path="./test_conf", config_name="test_conf")(_main)
    dora_main.dora.dir = tmpdir
    return dora_main


def call(main, argv):
    old_argv = list(sys.argv)
    try:
        sys.argv[1:] = argv
        main()
    finally:
        sys.argv = old_argv
    return _ret


def test_hydra_git_save(tmpdir):
    _main.__module__ = __name__
    main = get_main(tmpdir)
    argv = ["optim.loss=git_save"]
    xp = main.get_xp(argv)
    main.init_xp(xp)
    xp.dora.git_save = True

    clone = get_new_clone(main)
    assign_clone(xp, clone)
    with enter_clone(clone):
        call(main, argv)


def test_hydra(tmpdir):
    _main.__module__ = __name__
    main = get_main(tmpdir)
    xp = call(main, [])
    assert isinstance(xp, XP)
    assert len(xp.sig) > 0

    assert main.get_slurm_config().cpus_per_task == 5

    argv = ["num_workers=40"]
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig
    assert xp2.cfg.num_workers == 40

    argv = main.value_to_argv({"useless.a": 3})
    assert len(argv) > 0
    xp2 = call(main, argv)
    assert xp.sig == xp2.sig
    assert xp2.cfg.useless.a == 3

    pre = ["useless.b=false", "optim.loss=l1"]
    argv = main.value_to_argv(pre)
    assert argv == pre
    xp2 = call(main, pre)
    assert xp.sig != xp2.sig

    assert argv == main.get_argv_from_sig(xp2.sig)

    xp3 = main.get_xp_from_sig(xp2.sig)
    assert xp2.argv == xp3.argv
    assert xp2.delta == xp3.delta
    assert xp2.sig == xp3.sig
    assert xp2.dora == xp3.dora

    metrics = main.get_xp_history(xp3)
    assert metrics[-1]["loss"] == 0.1

    name = main.get_name(xp3)
    assert name == "opt.loss=l1"

    argv = ["+k=youpi"]
    xp2 = call(main, argv)
    assert xp2.cfg.k == "youpi"

    with pytest.raises(ValueError):
        main.value_to_argv(0.5)

    argv = ["plop.b=5"]
    xp2 = call(main, argv)
    assert xp2.cfg.plop.b == 5
    assert not hasattr(xp2.cfg, "lapin")

    argv = ["group=lapin"]
    xp2 = call(main, argv)
    assert xp2.cfg.lapin.a == 5
    assert not hasattr(xp2.cfg, "plop")

    argv = ["group=lapin", "plop.b=5"]
    with pytest.raises(ConfigCompositionException):
        xp2 = call(main, argv)


def test_complex_types(tmpdir):
    # Test complex types parsing (e.g. lists and dict)
    _main.__module__ = __name__
    main = get_main(tmpdir)
    xp = call(main, [])
    print(xp.cfg.complex)
    assert xp.cfg.complex.a == [1, 2, 3]
    xp = call(main, ["complex.a=[0]"])
    assert xp.cfg.complex.a == [0]
    xp = call(main, ["complex.b.a=50"])
    assert xp.cfg.complex.b == {"a": 50, "b": 2}
    xp = call(main, ["complex.b={a:21}"])
    assert xp.cfg.complex.b == {"a": 21, "b": 2}
    argv = main.value_to_argv({"complex.b": {"a": 21, "b": 52}})
    xp = call(main, argv)
    assert xp.cfg.complex.b == {"a": 21, "b": 52}


def test_config_groups_unaffected_by_no_copy(tmpdir):
    """`_get_config_groups` suppresses Hydra's internal deepcopies for speed.

    That is only sound if it does not change the answer, so compare against a
    run with Hydra's real `__deepcopy__` in place.
    """
    # `hydra_main` rewrites `_main.__module__`, and the config path is resolved
    # relative to it, so restore it the way the other tests here do.
    _main.__module__ = __name__
    main = get_main(tmpdir)
    assert main._get_config_groups(fast=True) == main._get_config_groups(fast=False)


def test_get_existing_xp_reads_what_the_run_stored(tmpdir):
    """The stored config wins over recomposition.

    An experiment outlives its config files, so reading back what it actually
    ran with has to be possible even once the tree has moved on.
    """
    import yaml

    _main.__module__ = __name__
    main = get_main(tmpdir)
    argv = ["optim.loss=stored"]
    xp = main.get_xp(argv)
    main.init_xp(xp)

    # Stand in for a config tree that has since changed: a value no current
    # composition could produce.
    xp._hydra_config.parent.mkdir(parents=True, exist_ok=True)
    stored = {"optim": {"loss": "stored", "lr": 0.123}, "gone": "only-on-disk"}
    xp._hydra_config.write_text(yaml.safe_dump(stored))

    loaded = main.get_existing_xp_from_sig(xp.sig)
    assert loaded.sig == xp.sig
    assert loaded.argv == list(argv)
    assert loaded.cfg.gone == "only-on-disk"
    assert loaded.cfg.optim.lr == 0.123
    # init_xp persisted the delta, so the name survives without recomposing.
    assert loaded.delta == xp.delta
    assert main.get_name(loaded) == main.get_name(xp)


def test_get_existing_xp_without_delta_falls_back_to_sig(tmpdir):
    """Experiments created before the delta was persisted keep working; they
    just lose their name, which is not worth recomposing a config tree for."""
    _main.__module__ = __name__
    main = get_main(tmpdir)
    xp = main.get_xp(["optim.loss=nodelta"])
    main.init_xp(xp)
    xp._delta_cache.unlink()

    loaded = main.get_existing_xp_from_sig(xp.sig)
    assert loaded.delta is None
    assert main.get_name(loaded) == xp.sig


def test_config_comparisons_do_not_share_default_paths():
    from omegaconf import OmegaConf

    from ..hydra import _compare_config

    reference = OmegaConf.create({"outer": {"value": 1}})
    changed = OmegaConf.create({"outer": {"value": 2}})
    first = _compare_config(reference, changed)
    assert next(first).path == ["outer", "value"]
    # Starting another comparison while the first is suspended must not reuse its path.
    assert next(_compare_config(reference, changed)).path == ["outer", "value"]
    first.close()
