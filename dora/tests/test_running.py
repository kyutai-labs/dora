"""Live job accounting, persisted membership, and human/script renderings."""

import json
import subprocess
from argparse import Namespace
from unittest.mock import Mock

import pytest

from dora import running
from dora.__main__ import get_parser
from dora.conf import DoraConfig

from .test_inspect import make_xp


def args(**kwargs):
    return Namespace(**{"limit": None, "json": True, **kwargs})


@pytest.fixture
def project(tmp_path, monkeypatch):
    dora = DoraConfig(dir=tmp_path)
    first = make_xp(dora, "aaaaaaaa", ["model=big", "seed=1"], job_id="100_0")
    second = make_xp(dora, "bbbbbbbb", ["model=big", "seed=2"], job_id="100_1")
    third = make_xp(dora, "cccccccc", ["model=small"], job_id="200")
    make_xp(dora, "dddddddd", job_id="300")  # Finished: absent from squeue.
    (third / "job.json").write_text(
        json.dumps({"job_id": "200", "dependent_job_ids": ["201"], "array_job_ids": ["999"]})
    )
    grid = dora.dir / dora._grids / "some.grid"
    grid.mkdir(parents=True)
    for folder in (first, second):
        (grid / folder.name).symlink_to(folder)
    by_id = dora.dir / dora.shep.by_id
    by_id.mkdir(parents=True)
    (by_id / "50").symlink_to(first)  # A still-running older attempt.
    monkeypatch.setattr(
        running,
        "running_jobs",
        lambda: [
            {"job": "100_0", "partition": "new", "gpus": 16},
            {"job": "100_1", "partition": "old", "gpus": 8},
            {"job": "201", "partition": "cpu", "gpus": 0},
            {"job": "50", "partition": "old", "gpus": 4},
        ],
    )
    return dora


def test_live_jobs_arrays_dependents_and_short_names(project, capsys):
    assert running.running_action(args(), project) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == result["shown"] == 4
    assert result["gpus"] == 28
    groups = {group["grid"]: group for group in result["grids"]}
    grid = groups["some.grid"]
    assert grid["gpus"] == 28
    assert "base_name" not in grid
    assert {job["name"] for job in grid["jobs"]} == {"seed=1", "seed=2"}
    assert {job["job"] for job in grid["jobs"]} == {"50", "100_0", "100_1"}
    assert groups[None]["jobs"][0]["job"] == "201"


def test_shared_membership_does_not_inflate_overall_totals(project, capsys):
    grid = project.dir / project._grids / "other.grid"
    grid.mkdir()
    (grid / "aaaaaaaa").symlink_to(project.dir / project.xps / "aaaaaaaa")
    assert running.running_action(args(), project) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["gpus"] == 28
    assert result["count"] == 4
    groups = {group["grid"]: group for group in result["grids"]}
    assert groups["other.grid"]["gpus"] == 20
    assert groups["some.grid"]["gpus"] == 28
    assert groups["other.grid"]["jobs"][0]["grids"] == ["other.grid", "some.grid"]


def test_limit_preserves_totals_and_names(project, capsys):
    running.running_action(args(limit=2), project)
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 4 and result["shown"] == 2 and result["gpus"] == 28
    assert sum(group["shown"] for group in result["grids"]) == 2
    assert result["grids"][1]["jobs"][0]["name"] == "seed=1"


@pytest.mark.parametrize("compact", [False, True])
def test_compact_and_non_tty_are_plain_tables(project, capsys, compact):
    running.running_action(args(json=False, compact=compact), project)
    output = capsys.readouterr().out
    assert "\x1b" not in output
    lines = output.splitlines()
    assert lines[0] == "4 running job(s), 28 GPUs"
    assert f"some.grid [{project.dir}]: 3 job(s), 28 GPUs" in lines
    jobs = [
        line.split()
        for line in lines
        if line.strip().startswith(("aaaaaaaa", "bbbbbbbb", "cccccccc"))
    ]
    assert len(jobs) == 4
    assert {row[1] for row in jobs} == {"50", "100_0", "100_1", "201"}
    assert sum(int(row[2]) for row in jobs) == 28


def test_pretty_grouped_table(project, capsys):
    running.running_action(args(json=False, pretty=True), project)
    output = capsys.readouterr().out
    assert f"some.grid [{project.dir}]: 3 job(s), 28 GPUs" in output
    assert "common:" not in output and "model=big" not in output
    assert "seed=1" in output and "seed=2" in output
    assert "4 running job(s), 28 GPUs" in output
    assert "(no grid)" in output


@pytest.mark.parametrize(
    "tres, expected",
    [
        ("cpu=16,node=2,gres/gpu=16,gres/gpu:a100=16", 16),
        ("cpu=8,gres/gpu:a100=2,gres/gpu:h100=4", 6),
        ("cpu=4,node=1", 0),
        ("N/A", None),
        ("", None),
        ("gres/gpu=oops", None),
    ],
)
def test_gpu_allocations(tres, expected):
    assert running.gpu_count(tres) == expected


def test_scheduler_query_expands_arrays_and_parses_allocations(monkeypatch):
    proc = Mock(
        stdout="100_0|learn|cpu=128,node=2,gres/gpu=16|/logs/a|/code|train\n"
        "42|cpu|cpu=1|/logs/b|/code|cpu-job\n"
    )
    run = Mock(return_value=proc)
    monkeypatch.setattr(running.sp, "run", run)
    assert running.running_jobs() == [
        {
            "job": "100_0",
            "partition": "learn",
            "gpus": 16,
            "stdout": "/logs/a",
            "workdir": "/code",
            "name": "train",
        },
        {
            "job": "42",
            "partition": "cpu",
            "gpus": 0,
            "stdout": "/logs/b",
            "workdir": "/code",
            "name": "cpu-job",
        },
    ]
    assert "--array" in run.call_args.args[0]
    assert "--states=RUNNING" in run.call_args.args[0]
    assert run.call_args.kwargs["timeout"] == 30


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("squeue"),
        subprocess.TimeoutExpired("squeue", 30),
        ValueError("Unexpected squeue output"),
    ],
)
def test_scheduler_failure_is_not_an_empty_success(project, monkeypatch, capsys, failure):
    monkeypatch.setattr(running, "running_jobs", Mock(side_effect=failure))
    assert running.running_action(args(), project) == 1
    captured = capsys.readouterr()
    assert "error" in json.loads(captured.out)
    assert "cannot list running jobs" in captured.err


def test_empty_and_unknown_allocations(project, monkeypatch, capsys):
    monkeypatch.setattr(running, "running_jobs", list)
    assert running.running_action(args(), project) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0
    monkeypatch.setattr(
        running, "running_jobs", lambda: [{"job": "100_0", "partition": "learn", "gpus": None}]
    )
    running.running_action(args(), project)
    result = json.loads(capsys.readouterr().out)
    assert result["unknown_gpus"] == 1 and result["gpus"] == 0
    assert result["grids"][0]["jobs"][0]["gpus"] is None


def test_running_cli():
    parsed = get_parser().parse_args(["running", "--compact", "--limit", "3"])
    assert parsed.action is running.running_action
    assert parsed.read_only and parsed.compact and parsed.limit == 3


def test_discover_across_repos_without_importing_training(tmp_path, monkeypatch, capsys):
    from dora import __main__ as cli

    jobs = []
    for name, job_id, gpus in [("project_a", "123_0", 8), ("project_b", "456", 16)]:
        config = DoraConfig(dir=tmp_path / name)
        folder = make_xp(config, "aaaaaaaa", [f"model={name}"], job_id=job_id)
        grid = config.dir / config._grids / "absent.on.this.branch"
        grid.mkdir(parents=True)
        (grid / folder.name).symlink_to(folder)
        stdout = (
            config.dir / "arrays" / "train_array" / "%A_%a_0_log.out"
            if "_" in job_id
            else folder / "submitit" / "%j_0_log.out"
        )
        jobs.append(
            {
                "job": job_id,
                "gpus": gpus,
                "partition": "learn",
                "stdout": str(stdout),
                "workdir": str(tmp_path / "deleted-worktree"),
                "name": "train",
            }
        )
    jobs.append({"job": "999", "gpus": 1, "partition": "learn", "name": "external-job"})
    monkeypatch.setattr(running, "running_jobs", lambda: jobs)
    # No project config in cwd and no training/grid Python code anywhere.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "get_main", Mock(side_effect=AssertionError("no training import")))
    monkeypatch.setattr(cli, "get_dora_config", Mock(side_effect=AssertionError("no cwd lookup")))
    monkeypatch.setattr("sys.argv", ["dora", "running", "--json"])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 3 and result["gpus"] == 25
    assert len(result["grids"]) == 3  # Same grid name and sig in two roots stay distinct.
    identified = [g for g in result["grids"] if g["root"]]
    assert {g["gpus"] for g in identified} == {8, 16}
    assert all(g["grid"] == "absent.on.this.branch" for g in identified)
    unknown = next(g for g in result["grids"] if g["root"] is None)
    assert unknown["jobs"][0]["name"] == "external-job"
    assert unknown["jobs"][0]["sig"] is None


def test_array_siblings_are_not_mapped_to_the_wrong_xp(project):
    jobs = [{"job": "999"}]
    assert running._job_targets(project, jobs) == {}


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_does_not_query_scheduler(project, monkeypatch, limit):
    monkeypatch.setattr(running, "running_jobs", Mock(side_effect=AssertionError("invalid limit")))
    assert running.running_action(args(limit=limit), project) == 1


def test_pretty_wraps_full_names_using_local_toml(project, monkeypatch, capsys):
    monkeypatch.chdir(project.dir)
    (project.dir / "dora.toml").write_text("[dora]\nname_width = 16\n")
    name = "override=" + "abcdefghij" * 10
    common = "common=" + "0123456789" * 15
    monkeypatch.setattr(
        running, "_xp_names", lambda _, targets: ({t.sig: name for t in targets}, common)
    )
    running.running_action(args(json=False, pretty=True), project)
    output = capsys.readouterr().out
    assert "common:" not in output and "common=" not in output
    # Every piece, including the tail of an unbroken value, is retained.
    assert all(part in output for part in running.format_name(name, "pretty", 16).splitlines())
    running.running_action(args(json=False, compact=True), project)
    compact = capsys.readouterr().out
    assert "common=" not in compact and "base_name" not in compact
    shortened = running.format_name(name, "compact")
    assert len(shortened) <= 44 and shortened != name
    assert compact.count(shortened) == 4
    # A long name occupies exactly one row per job, without continuation lines.
    rows = [line for line in compact.splitlines() if line.startswith("  ")]
    assert len(rows) == 6  # four jobs and two column headers
    running.running_action(args(), project)
    result = json.loads(capsys.readouterr().out)
    assert result["grids"][0]["jobs"][0]["name"] == name
