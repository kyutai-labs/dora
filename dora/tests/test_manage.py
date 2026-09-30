"""Persisted Slurm settings and targeted status actions, using a fake scheduler."""

from argparse import Namespace
from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest
import submitit

from dora import _utils, inspect, manage
from dora.__main__ import get_parser
from dora.conf import SlurmConfig
from dora.shep import Shepherd, _JobArray
from .fake_shep import FakeJob, mock_shep
from .test_main import get_main


@pytest.fixture
def launched(tmp_path, monkeypatch):
    with mock_shep():
        main = get_main(tmp_path)
        shepherd = Shepherd(main)
        calls = []
        monkeypatch.setattr(_utils, "get_main", Mock(return_value=main))
        monkeypatch.setattr(
            manage, "job_states", lambda ids: {job: FakeJob.watcher.jobs[job] for job in ids}
        )

        def cancel(cmd, **kwargs):
            assert cmd[0] == "scancel"
            calls.append(cmd)
            for job in cmd[1:]:
                FakeJob.watcher.jobs[job] = "CANCELLED"

        monkeypatch.setattr(manage.sp, "run", cancel)
        yield main, shepherd, calls


def launch(main, shepherd, argv=(), **kwargs):
    sheep = shepherd.get_sheep_from_argv(argv)
    slurm = SlurmConfig(**kwargs)
    shepherd._submit(_JobArray(slurm, [sheep]))
    return sheep, slurm


def args(*targets, **kwargs):
    values = dict(
        targets=list(targets),
        cancel=False,
        restart=False,
        dry_run=False,
        json=True,
        limit=None,
        main_module=None,
        package=None,
    )
    values.update(kwargs)
    return Namespace(**values)


def test_snapshot_covers_arrays_dependents_and_last_success(launched, monkeypatch):
    main, shepherd, _ = launched
    first, slurm = launch(
        main,
        shepherd,
        gpus=16,
        partition="custom",
        dependents=2,
        setup=["module load cuda"],
        srun_args=["--container-image=image"],
        container_chdir=True,
        nodelist=["node1", "node2"],
    )
    saved = first.xp.folder / "slurm.json"
    assert json.loads(saved.read_text()) == asdict(slurm)
    assert slurm.srun_args == ["--container-image=image"]
    assert manage.load_slurm_config(first.xp.folder, main.dora) == slurm
    assert len(json.loads(first._json_job_file.read_text())["dependent_job_ids"]) == 2

    second = shepherd.get_sheep_from_argv(["--a=2"])
    slurm.dependents = 0
    slurm.partition = "new"
    shepherd._submit(_JobArray(slurm, [first, second]))
    assert json.loads(saved.read_text()) == asdict(slurm)
    assert json.loads((second.xp.folder / "slurm.json").read_text()) == asdict(slurm)
    previous = saved.read_text()
    monkeypatch.setattr(
        submitit.SlurmExecutor, "submit", Mock(side_effect=RuntimeError("submission failed"))
    )
    slurm.partition = "will-fail"
    with pytest.raises(RuntimeError, match="submission failed"):
        shepherd._submit(_JobArray(slurm, [first]))
    assert saved.read_text() == previous


def test_old_job_json_fallback_and_invalid_snapshot(launched):
    main, shepherd, _ = launched
    sheep, slurm = launch(main, shepherd, gpus=8, partition="old")
    snapshot = sheep.xp.folder / "slurm.json"
    snapshot.unlink()
    assert manage.load_slurm_config(sheep.xp.folder, main.dora) == slurm
    snapshot.write_text("broken")
    with pytest.raises(ValueError, match="Cannot read"):
        manage.load_slurm_config(sheep.xp.folder, main.dora)
    snapshot.write_text('{"unknown": 1}')
    with pytest.raises(ValueError, match="invalid Slurm"):
        manage.load_slurm_config(sheep.xp.folder, main.dora)


def test_cancel_only_selected_array_member_and_dependents(launched):
    main, shepherd, calls = launched
    first = shepherd.get_sheep_from_argv(["--a=1"])
    other = shepherd.get_sheep_from_argv(["--a=2"])
    shepherd._submit(_JobArray(SlurmConfig(), [first, other]))
    _utils.get_main.side_effect = AssertionError("Cancellation must not import training")
    assert inspect.status_action(args(first.xp.sig, cancel=True), main.dora) == 0
    assert calls == [["scancel", first.job.job_id]]
    assert other.state() != "CANCELLED"
    dependent, _ = launch(main, shepherd, ["--a=3"], dependents=2)
    assert inspect.status_action(args(dependent.xp.sig, cancel=True), main.dora) == 0
    assert calls[-1] == [
        "scancel",
        dependent.job.job_id,
        *[job.job_id for job in dependent._dependent_jobs],
    ]


def test_restart_uses_saved_params_argv_and_keeps_checkpoint(launched, monkeypatch, capsys):
    main, shepherd, calls = launched
    sheep, slurm = launch(
        main,
        shepherd,
        ["--a=5"],
        gpus=8,
        partition="saved",
        python="uv run --locked --no-editable python",
        dependents=1,
    )
    previous_job = sheep.job.job_id
    checkpoint = sheep.xp.folder / "checkpoint.th"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(main, "get_slurm_config", Mock(side_effect=AssertionError("use saved")))
    monkeypatch.setattr(Shepherd, "_check_orphans", Mock(side_effect=AssertionError("unrelated")))
    assert inspect.status_action(args(sheep.xp.sig, restart=True), main.dora) == 0
    record = json.loads(capsys.readouterr().out)["experiments"][0]
    assert record["job"] != previous_job
    assert calls == [["scancel", previous_job, sheep._dependent_jobs[0].job_id]]
    assert checkpoint.read_bytes() == b"checkpoint"
    assert manage.load_slurm_config(sheep.xp.folder, main.dora) == slurm
    assert json.loads((sheep.xp.folder / ".argv.json").read_text()) == ["--a=5"]
    _utils.get_main.assert_called_once()


def test_restart_from_old_metadata_and_completed_job(launched):
    main, shepherd, calls = launched
    sheep, slurm = launch(main, shepherd, gpus=4)
    (sheep.xp.folder / "slurm.json").unlink()
    sheep.job._state = "COMPLETED"
    assert inspect.status_action(args(sheep.xp.sig, restart=True), main.dora) == 0
    assert calls == []
    assert manage.load_slurm_config(sheep.xp.folder, main.dora) == slurm


@pytest.mark.parametrize("bad", ["signature", "missing_slurm", "invalid_argv", "unknown_target"])
def test_preflight_prevents_partial_actions(launched, bad):
    main, shepherd, calls = launched
    first, _ = launch(main, shepherd, ["--a=1"])
    second, _ = launch(main, shepherd, ["--a=2"])
    targets = [first.xp.sig, second.xp.sig]
    if bad == "signature":
        (second.xp.folder / ".argv.json").write_text('["--a=999"]')
    elif bad == "invalid_argv":
        (second.xp.folder / ".argv.json").write_text("{}")
    elif bad == "missing_slurm":
        (second.xp.folder / "slurm.json").unlink()
        second._json_job_file.write_text(json.dumps({"job_id": second.job.job_id}))
    else:
        targets.append("missing")
    assert inspect.status_action(args(*targets, restart=True), main.dora) == 1
    assert calls == []
    assert len(FakeJob.watcher.jobs) == 2


@pytest.mark.parametrize("action", ["cancel", "restart"])
def test_dry_run_does_not_cancel_or_submit(launched, action):
    main, shepherd, calls = launched
    sheep, _ = launch(main, shepherd)
    assert inspect.status_action(args(sheep.xp.sig, dry_run=True, **{action: True}), main.dora) == 0
    assert calls == []
    assert len(FakeJob.watcher.jobs) == 1


def test_grid_targets_read_membership_and_limit_only_output(launched, capsys):
    main, shepherd, calls = launched
    grid = main.dora.dir / main.dora._grids / "missing.grid.module"
    grid.mkdir(parents=True)
    for a in range(3):
        sheep, _ = launch(main, shepherd, [f"--a={a}"])
        (grid / sheep.xp.sig).symlink_to(sheep.xp.folder)
    assert inspect.status_action(args("missing.grid.module", cancel=True, limit=1), main.dora) == 0
    assert len(calls[0]) == 4
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 3 and result["shown"] == 1


def test_cli_actions_are_explicit_and_exclusive():
    parser = get_parser()
    plain = parser.parse_args(["status", "12345678"])
    assert not plain.cancel and not plain.restart
    restart = parser.parse_args(["status", "12345678", "--restart", "--dry-run"])
    assert restart.restart and restart.dry_run
    with pytest.raises(SystemExit):
        parser.parse_args(["status", "12345678", "--restart", "--cancel"])


def test_git_save_preparation_failure_does_not_cancel(launched, monkeypatch):
    from dora import git_save

    main, shepherd, calls = launched
    sheep, _ = launch(main, shepherd)
    main.dora.git_save = True
    monkeypatch.setattr(
        git_save, "get_new_clone", Mock(side_effect=RuntimeError("clone setup failed"))
    )
    assert inspect.status_action(args(sheep.xp.sig, restart=True), main.dora) == 1
    assert calls == []
    assert len(FakeJob.watcher.jobs) == 1


def test_cancellation_failure_does_not_submit(launched, monkeypatch):
    main, shepherd, _ = launched
    sheep, _ = launch(main, shepherd)
    monkeypatch.setattr(
        manage.sp, "run", Mock(side_effect=manage.sp.CalledProcessError(1, "scancel"))
    )
    assert inspect.status_action(args(sheep.xp.sig, restart=True), main.dora) == 1
    assert len(FakeJob.watcher.jobs) == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_restart_partition_override(launched, capsys, dry_run):
    main, shepherd, calls = launched
    sheep, slurm = launch(main, shepherd, gpus=16, partition="old", dependents=1)
    snapshot = (sheep.xp.folder / "slurm.json").read_bytes()
    assert (
        inspect.status_action(
            args(sheep.xp.sig, restart=True, partition="new", dry_run=dry_run), main.dora
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["experiments"][0]["partition"] == "new"
    if dry_run:
        assert not calls
        assert (sheep.xp.folder / "slurm.json").read_bytes() == snapshot
    else:
        slurm.partition = "new"
        assert manage.load_slurm_config(sheep.xp.folder, main.dora) == slurm


@pytest.mark.parametrize("flag", ["-p", "--partition"])
def test_restart_partition_parser(flag):
    parsed = get_parser().parse_args(["status", "12345678", "--restart", flag, "new"])
    assert parsed.partition == "new"


@pytest.mark.parametrize("action", [[], ["--cancel"]])
def test_partition_requires_restart(monkeypatch, action):
    from dora.__main__ import main

    monkeypatch.setattr("sys.argv", ["dora", "status", "12345678", "-p", "new", *action])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
