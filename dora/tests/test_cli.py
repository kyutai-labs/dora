# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the command line surface itself."""

import pytest

from ..__main__ import get_parser


@pytest.fixture(scope="module")
def parser():
    return get_parser()


@pytest.mark.parametrize(
    "argv, attr, expected",
    [
        # The one that motivated this: guessing wrong on a flag whose whole purpose
        # is to not do the thing is a bad way to find out.
        (["grid", "g", "--dry-run"], "dry_run", True),
        (["grid", "g", "--dry_run"], "dry_run", True),
        (["status", "s", "--cancel", "--dry-run"], "dry_run", True),
        (["status", "s", "--cancel", "--dry_run"], "dry_run", True),
        # ... and every other multiword flag, in both spellings.
        (["grid", "g", "--no-monitoring"], "monitor", False),
        (["grid", "g", "--no_monitoring"], "monitor", False),
        (["grid", "g", "--replace-done"], "replace_done", True),
        (["grid", "g", "--no-git-save"], "git_save", False),
        (["info", "--from-sig", "abc"], "from_sig", "abc"),
        (["info", "--from_sig=abc"], "from_sig", "abc"),
        (["info", "--from-sig=abc"], "from_sig", "abc"),
        (["info", "--job-id", "42"], "job_id", "42"),
        (["run", "--ddp-workers", "4"], "ddp_workers", 4),
    ],
)
def test_both_separators_are_accepted(parser, argv, attr, expected):
    assert getattr(parser.parse_args(argv), attr) == expected


def test_help_shows_one_spelling_per_flag(parser):
    """The aliases resolve when parsing but stay out of the actions, so help
    does not list every flag twice."""
    text = parser.format_help()
    for action in parser._actions:
        if action.dest == "command":
            for sub in action.choices.values():
                text += sub.format_help()
    assert "--dry_run" in text
    assert "--dry-run" not in text.replace("--dry-run requires", "")


def test_a_typo_is_still_a_typo(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["grid", "g", "--dry-runn"])
