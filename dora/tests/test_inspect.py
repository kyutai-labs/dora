# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the read-only inspection commands.

The point of these commands is that their output is cheap and safe to read
programmatically, so most of what is worth testing is the output discipline:
caps, no colour, no pathological lines.
"""
import json
import time as _time

import pytest

from .. import inspect
from ..conf import DoraConfig


@pytest.fixture
def dora(tmp_path):
    return DoraConfig(dir=tmp_path)


def make_xp(dora, sig, argv=None, history=None, job_id=None):
    folder = dora.dir / dora.xps / sig
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".argv.json").write_text(json.dumps(argv or []))
    if history is not None:
        (folder / "history.json").write_text(json.dumps(history))
    if job_id is not None:
        (folder / "job.json").write_text(json.dumps({"job_id": job_id}))
    return folder


def test_all_digit_signature_is_not_a_job_id(dora):
    """Signatures are eight hex characters, which can be all digits. Reading
    one as a Slurm job id sends the lookup somewhere else entirely."""
    make_xp(dora, "16076961")
    resolved = inspect.resolve_targets(["16076961"], dora)
    assert not resolved.problems
    assert [t.sig for t in resolved.targets] == ["16076961"]


def test_unknown_target_is_reported_not_raised(dora):
    resolved = inspect.resolve_targets(["nope"], dora)
    assert resolved.targets == []
    assert "nope" in resolved.problems[0]


def test_grid_expands_to_its_experiments(dora):
    make_xp(dora, "aaaaaaaa")
    make_xp(dora, "bbbbbbbb")
    grid = dora.dir / dora._grids / "some.grid"
    grid.mkdir(parents=True)
    for sig in ("aaaaaaaa", "bbbbbbbb"):
        (grid / sig).symlink_to(dora.dir / dora.xps / sig)
    resolved = inspect.resolve_targets(["some.grid"], dora)
    assert sorted(t.sig for t in resolved.targets) == ["aaaaaaaa", "bbbbbbbb"]
    assert all(t.grid == "some.grid" for t in resolved.targets)


def test_duplicate_targets_are_shown_once(dora):
    make_xp(dora, "aaaaaaaa")
    resolved = inspect.resolve_targets(["aaaaaaaa", "@aaaaaaaa"], dora)
    assert len(resolved.targets) == 1


def test_history_is_downsampled_not_dumped(dora, capsys):
    """A real history is hundreds of KB; printing it defeats the purpose."""
    history = [{"valid": {"ce": 1.0 / (i + 1), "noise": i}} for i in range(500)]
    make_xp(dora, "cccccccc", history=history)

    args = _Args(targets=["cccccccc"], stage="valid", keys="ce", every=None,
                 limit=None, json=False)
    assert inspect.metrics_action(args, dora) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= inspect.MAX_METRIC_ROWS + 2
    assert "noise" not in out
    assert "500 epochs" in out


def test_every_samples_across_the_whole_run(dora, capsys):
    history = [{"valid": {"ce": float(i)}} for i in range(100)]
    make_xp(dora, "dddddddd", history=history)
    args = _Args(targets=["dddddddd"], stage="valid", keys="ce", every=25,
                 limit=None, json=False)
    inspect.metrics_action(args, dora)
    epochs = [line.split()[0] for line in capsys.readouterr().out.splitlines()[2:]]
    assert epochs == ["1", "26", "51", "76"]


def test_log_strips_colour_and_cuts_giant_lines(dora, capsys):
    """Hydra prints every override on one line on error; on a real project that
    is ~4 KB, which is around a thousand tokens of nothing."""
    folder = make_xp(dora, "eeeeeeee")
    (folder / "submitit").mkdir()
    (folder / "submitit" / "1_0_log.out").write_text(
        "\x1b[36mcoloured\x1b[0m\n" + "x" * 5000 + "\n")

    args = _Args(targets=["eeeeeeee"], limit=10, grep=None, rank=None, job=None,
                 json=False)
    assert inspect.log_action(args, dora) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "coloured" in out
    assert max(len(ln) for ln in out.splitlines()) < inspect.MAX_LINE_CHARS
    assert "+4800 chars" in out


def test_why_reports_the_cause_once_across_ranks(dora, capsys):
    """Eight ranks dying of the same thing is one problem, not eight."""
    folder = make_xp(dora, "ffffffff", job_id="42")
    (folder / "submitit").mkdir()
    for rank in range(8):
        (folder / "submitit" / f"42_{rank}_log.out").write_text(
            f"[2026-01-0{rank}] step\ntorch.OutOfMemoryError: CUDA out of memory\n")

    args = _Args(targets=["ffffffff"], job=None, json=False, limit=None, attempts=3)
    assert inspect.why_action(args, dora) == 0
    out = capsys.readouterr().out
    assert "out of memory" in out
    assert out.count("out of memory") <= 3
    assert "other rank" in out


def test_output_is_capped(capsys):
    inspect.emit(["x" * 200] * 200)
    out = capsys.readouterr().out
    assert len(out) <= inspect.MAX_TOTAL_CHARS + 100
    assert "truncated" in out


def test_elide_parts_keeps_whole_key_values():
    """Cutting through the middle destroys exactly the part that distinguishes
    one experiment from another, since shared parts come first."""
    name = "lang=en model=big lr=1e-4 seed=1"
    assert inspect.elide_parts(name, 100) == name
    short = inspect.elide_parts(name, 20)
    assert short.startswith("lang=en")
    assert "=" in short and short.endswith("+2")


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_traceback_stops_at_the_exception(dora, capsys):
    """A distributed job prints hundreds of lines of NCCL teardown after it
    dies. Running the traceback to end-of-file buries the one line anyone
    wanted."""
    folder = make_xp(dora, "11111111", job_id="7")
    (folder / "submitit").mkdir()
    (folder / "submitit" / "7_0_log.out").write_text(
        "Error executing job with overrides: " + "x=1 " * 900 + "\n"
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in main\n'
        "    boom()\n"
        "ZeroDivisionError: float division by zero\n"
        + "\n".join(f"NCCL INFO teardown line {i}" for i in range(200)) + "\n")

    args = _Args(targets=["11111111"], job=None, json=False, limit=None, attempts=3)
    assert inspect.why_action(args, dora) == 0
    out = capsys.readouterr().out
    assert "ZeroDivisionError: float division by zero" in out
    assert "NCCL INFO teardown" not in out
    # Hydra wraps every exception in main with that banner, so trusting it
    # would report a division by zero as a configuration error.
    assert "hydra config error" not in out
    assert "x=1 x=1" not in out


def test_why_looks_back_through_earlier_attempts(dora, capsys):
    """The current job succeeding does not mean nothing went wrong: an
    experiment is usually requeued or resubmitted several times, and the
    failure people ask about is often in an earlier attempt."""
    folder = make_xp(dora, "22222222", job_id="900")
    (folder / "submitit").mkdir()
    (folder / "submitit" / "400_0_log.out").write_text(
        "Traceback (most recent call last):\n"
        '  File "train.py", line 1, in main\n'
        "FileNotFoundError: no such data\n")
    (folder / "submitit" / "900_0_log.out").write_text("all good\ndone\n")
    import os
    now = _time.time()
    os.utime(folder / "submitit" / "400_0_log.out", (now - 500, now - 500))
    os.utime(folder / "submitit" / "900_0_log.out", (now, now))

    args = _Args(targets=["22222222"], job=None, json=False, limit=None, attempts=3)
    assert inspect.why_action(args, dora) == 0
    out = capsys.readouterr().out
    assert "FileNotFoundError: no such data" in out
    assert "attempt 400" in out
    assert "not the current job 900" in out


def test_current_attempt_comes_from_job_json_not_the_biggest_id(dora):
    """Slurm's accounting database gets reset and job ids start again from a
    low number, so a larger id is not a later job."""
    folder = make_xp(dora, "33333333", job_id="12")
    (folder / "submitit").mkdir()
    for job in ("12", "99999"):
        (folder / "submitit" / f"{job}_0_log.out").write_text("x\n")
    target = inspect.Target(sig="33333333", folder=folder)
    assert log_attempt_names(target, current_job="12")[0] == "12"


def log_attempt_names(target, current_job=None):
    return [name for name, _ in inspect.log_attempts(target, current_job=current_job)]


def test_mode_follows_stdout_when_no_flag_is_given(monkeypatch):
    """Neither audience should have to remember a flag: a terminal gets the
    readable rendering, a pipe gets the one a script can consume."""
    class _Out:
        def __init__(self, tty):
            self._tty = tty

        def isatty(self):
            return self._tty

    monkeypatch.setattr(inspect.sys, "stdout", _Out(True))
    assert inspect.output_mode(_Args()) == inspect.PRETTY
    monkeypatch.setattr(inspect.sys, "stdout", _Out(False))
    assert inspect.output_mode(_Args()) == inspect.COMPACT


def test_explicit_flags_beat_the_tty(monkeypatch):
    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(inspect.sys, "stdout", _Tty())
    assert inspect.output_mode(_Args(compact=True)) == inspect.COMPACT
    assert inspect.output_mode(_Args(json=True)) == inspect.JSON
    # json wins over compact: it is the more specific request.
    assert inspect.output_mode(_Args(compact=True, json=True)) == inspect.JSON


def test_no_color_is_respected(monkeypatch):
    """no-color.org: an environment variable users already know."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert inspect.use_colour(inspect.PRETTY)
    monkeypatch.setenv("NO_COLOR", "1")
    assert not inspect.use_colour(inspect.PRETTY)
    assert not inspect.use_colour(inspect.COMPACT)


def test_columns_line_up_even_when_coloured(monkeypatch):
    """Padding has to count visible characters, or every colour code shifts
    the column after it."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    plain = inspect.columns([["ab", "c"], ["d", "ef"]], ["h1", "h2"], inspect.COMPACT)
    coloured = inspect.columns(
        [[inspect.paint("ab", "31", inspect.PRETTY), "c"], ["d", "ef"]],
        ["h1", "h2"], inspect.PRETTY)
    assert [inspect.ANSI_RE.sub("", ln) for ln in coloured] == plain


def test_log_keeps_colour_only_when_pretty(dora, capsys, monkeypatch):
    """Solver logs are colourised, which is what a person reading them wants
    and what anything parsing them does not."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    folder = make_xp(dora, "77777777")
    (folder / "submitit").mkdir()
    (folder / "submitit" / "1_0_log.out").write_text("\x1b[36mhello\x1b[0m\n")

    base = dict(targets=["77777777"], limit=5, grep=None, rank=None, job=None)
    inspect.log_action(_Args(**base, compact=True), dora)
    assert "\x1b" not in capsys.readouterr().out

    inspect.log_action(_Args(**base, pretty=True), dora)
    assert "\x1b[36m" in capsys.readouterr().out

    inspect.log_action(_Args(**base, json=True), dora)
    payload = json.loads(capsys.readouterr().out)
    assert payload["lines"] == ["hello"]      # never escape codes in JSON
