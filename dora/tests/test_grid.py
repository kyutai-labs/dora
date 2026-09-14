# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..conf import SubmitRules
from ..explore import Explorer, Launcher
from ..hydra import HydraMain
from ..grid import run_grid, RunGridArgs
from .fake_shep import mock_shep
from .test_main import get_main
from .test_hydra import get_main as get_main_hydra

_ret = None


def explore_1(launcher: Launcher):
    launcher()
    launcher(num_workers=42)


def explore_2(launcher: Launcher):
    launcher(num_workers=42, a=4)


def test_shep(tmpdir):
    def rgrid(explore):
        return run_grid(main, Explorer(explore), "unittest", slurm=slurm, rules=rules, args=args)

    with mock_shep():
        main = get_main(tmpdir)
        slurm = main.get_slurm_config()
        rules = SubmitRules()
        args = RunGridArgs()

        args.monitor = False
        args.dry_run = True

        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].job is None

        args.dry_run = False
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].job.job_id == "0"
        assert not sheeps[0].is_done()

        args.cancel = True
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "CANCELLED"
        assert sheeps[0].is_done()

        args.cancel = False
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "CANCELLED"

        rules.retry = True
        sheeps = rgrid(explore_1)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "UNKNOWN"
        assert sheeps[0].job.job_id == "1"

        old_sheep = sheeps[0]

        args.verbose = True
        sheeps = rgrid(explore_2)
        assert len(sheeps) == 1
        assert sheeps[0].state() == "UNKNOWN"
        assert sheeps[0].job.job_id == "2"
        assert old_sheep.state() == "CANCELLED"


def explore_hydra(launcher: Launcher):
    launcher.bind_({"epochs": 50, "optim.loss": "123", "num_workers": None})
    launcher({"complex.a": [{"test": "weird"}]})
    launcher({"complex.b": {"a": 21, "b": 4}})
    launcher({"+complex.b": {"a": 21, "b": 4, "c": 13}})


def test_shep_hydra(tmpdir):
    def rgrid(explore):
        return run_grid(main, Explorer(explore), "unittest", rules=rules, args=args)

    HydraMain._slow = False
    with mock_shep():
        main = get_main_hydra(tmpdir)
        rules = SubmitRules()
        args = RunGridArgs()
        args.monitor = False
        args.dry_run = True

        sheeps = rgrid(explore_hydra)
        assert len(sheeps) == 3
        cfg = sheeps[0].xp.cfg
        assert cfg.epochs == 50
        assert cfg.optim.loss == "123"
        assert cfg.num_workers is None
        assert cfg.complex.a == [{"test": "weird"}]

        cfg = sheeps[1].xp.cfg
        assert cfg.complex.b == {"a": 21, "b": 4}

        cfg = sheeps[2].xp.cfg
        assert cfg.complex.b == {"a": 21, "b": 4, "c": 13}


def test_dry_run_writes_nothing(tmpdir):
    """A simulated run must leave the Dora directory byte-for-byte untouched.

    It used to create the grid folder before checking `dry_run`, and the
    Shepherd created its bookkeeping folders unconditionally, so a "simulation"
    left behind an empty grid that afterwards looks like one someone launched
    and cancelled.
    """

    def snapshot(root):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*"))

    with mock_shep():
        main = get_main(tmpdir)
        root = main.dora.dir
        root.mkdir(exist_ok=True, parents=True)
        before = snapshot(root)

        args = RunGridArgs(monitor=False, dry_run=True, silent=True)
        sheeps = run_grid(
            main,
            Explorer(explore_1),
            "unittest_dry",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=args,
        )

        assert sheeps, "the explorer should still resolve experiments"
        assert snapshot(root) == before, "dry run touched the Dora directory"


def test_dry_run_with_init_writes_only_xp_caches(tmpdir):
    """`--dry_run --init` is the documented way to register signatures without
    scheduling, so it must still write the per-XP caches -- and nothing else."""
    with mock_shep():
        main = get_main(tmpdir)
        root = main.dora.dir
        root.mkdir(exist_ok=True, parents=True)

        args = RunGridArgs(monitor=False, dry_run=True, init=True, silent=True)
        sheeps = run_grid(
            main,
            Explorer(explore_1),
            "unittest_dry_init",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=args,
        )

        written = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        expected = sorted(
            f"{main.dora.xps}/{sheep.xp.sig}/{name}"
            for sheep in sheeps
            for name in (".argv.json", ".delta.json")
        )
        assert written == expected
        # In particular, no grid folder and no Shepherd bookkeeping.
        assert not (root / main.dora._grids / "unittest_dry_init").exists()


def test_read_only_shepherd_refuses_to_commit(tmpdir):
    """Reading job state must not be able to cancel anything.

    `Shepherd.__init__` runs an orphan check that cancels Slurm jobs, so a
    caller that only wants to look at state needs a way to opt out.
    """
    import pytest
    from ..shep import Shepherd

    with mock_shep():
        main = get_main(tmpdir)
        shepherd = Shepherd(main, read_only=True)
        assert not (main.dora.dir / main.dora.shep.orphans).exists()
        with pytest.raises(RuntimeError, match="read_only"):
            shepherd.commit()


def test_compact_grid_reports_stale_experiments(tmpdir, capsys):
    """Launching an edited grid cancels the experiments it no longer produces,
    so the output has to say which those are before anyone launches."""
    with mock_shep():
        main = get_main(tmpdir)
        args = RunGridArgs(monitor=False, dry_run=False, silent=True)
        run_grid(
            main,
            Explorer(explore_1),
            "unittest_plan",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=args,
        )
        capsys.readouterr()

        # explore_2 produces a different XP, so everything explore_1 scheduled
        # is now stale.
        args = RunGridArgs(monitor=False, dry_run=True, compact=True)
        run_grid(
            main,
            Explorer(explore_2),
            "unittest_plan",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=args,
        )

        out = capsys.readouterr().out
        # `run_grid` decides by `Sheep.is_done()`, and these are not done, so
        # they are the ones a real launch would actually cancel.
        assert "would be CANCELLED" in out
        assert "\x1b" not in out


def test_json_grid_output_is_parseable(tmpdir, capsys):
    """--json has to put exactly one object on stdout, so the monitoring
    chatter that normally precedes the table must be suppressed."""
    import json as json_module

    with mock_shep():
        main = get_main(tmpdir)
        args = RunGridArgs(monitor=False, dry_run=True, json=True)
        run_grid(
            main,
            Explorer(explore_1),
            "unittest_json",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=args,
        )
        payload = json_module.loads(capsys.readouterr().out)
        # explore_1's two calls differ only in an excluded parameter, so they
        # share a signature and are one experiment.
        assert len(payload["experiments"]) == 1
        assert {"sig", "state", "metrics"} <= set(payload["experiments"][0])


def test_grid_records_its_metric_columns(tmpdir):
    """`dora grid` is the only place that knows what an Explorer displays,
    because it is the only place that evaluates the grid file. Writing it down
    lets `dora status` show the same columns without importing anything."""
    import json as json_module

    import treetable as tt

    from ..inspect import METRIC_SPEC_NAME, load_metric_spec

    class Metrics(Explorer):
        def get_grid_metrics(self):
            return [
                tt.group("train", [tt.leaf("loss", ".3f")]),
                tt.group("valid", [tt.leaf("ce"), tt.leaf("ppl")]),
            ]

    with mock_shep():
        main = get_main(tmpdir)
        run_grid(
            main,
            Metrics(explore_1),
            "unittest_spec",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=RunGridArgs(monitor=False, silent=True),
        )

        spec = main.dora.dir / main.dora._grids / "unittest_spec" / METRIC_SPEC_NAME
        assert json_module.loads(spec.read_text())["columns"] == [
            "train.loss",
            "valid.ce",
            "valid.ppl",
        ]
        assert load_metric_spec(main.dora, "unittest_spec") == [
            "train.loss",
            "valid.ce",
            "valid.ppl",
        ]


def test_dry_run_records_no_metric_spec(tmpdir):
    """It is still a write, so it stays on the far side of --dry_run."""
    from ..inspect import METRIC_SPEC_NAME

    with mock_shep():
        main = get_main(tmpdir)
        run_grid(
            main,
            Explorer(explore_1),
            "unittest_spec_dry",
            slurm=main.get_slurm_config(),
            rules=SubmitRules(),
            args=RunGridArgs(monitor=False, silent=True, dry_run=True),
        )
        grid_folder = main.dora.dir / main.dora._grids / "unittest_spec_dry"
        assert not (grid_folder / METRIC_SPEC_NAME).exists()
        assert not grid_folder.exists()
