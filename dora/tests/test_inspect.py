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

    args = _Args(targets=["ffffffff"], job=None, json=False, limit=None)
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
