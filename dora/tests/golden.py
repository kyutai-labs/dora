# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Golden signature corpus: a regression net for anything that touches config
parsing, delta computation or signature hashing.

A signature is `sha1(sorted(delta))[:8]` (see `dora.xp._get_sig`), and the delta
is derived from a full Hydra composition. That makes signatures exquisitely
sensitive: a change in how defaults lists are walked, how `+key=value` additions
collapse into subtrees, or when interpolations get resolved will silently move
every signature in a project, orphaning years of experiments.

This module pins the mapping for a real corpus of experiments so such a change
fails loudly instead. Usage::

    python -m dora.tests.golden build corpus.json          # from a project root
    python -m dora.tests.golden check corpus.json

An entry's `status` records what happens *today*, so the checker can distinguish
three outcomes: a signature that moved (a regression, always a failure), an
experiment that stopped resolving (also a failure), and one that started
resolving again (an improvement, reported but never a failure).
"""

import json
import logging
import re
import sys
import typing as tp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

# A signature is 8 hex chars (`dora.xp._get_sig`). Experiment folders that do not
# match were renamed by hand (backups such as `<sig>_from_be_careful`) and would
# otherwise show up as spurious drift.
SIG_RE = re.compile(r"^[0-9a-f]{8}$")


OK = "ok"
DRIFT = "drift"
ERROR = "error"


@dataclass
class Entry:
    sig: str
    argv: list[str]
    status: str
    got_sig: str | None = None
    error: str | None = None


@dataclass
class Report:
    ok: list[str] = field(default_factory=list)
    drifted: list[tuple[str, str]] = field(default_factory=list)
    broke: list[tuple[str, str]] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    still_error: list[str] = field(default_factory=list)
    known_drift: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.drifted or self.broke)

    def summary(self) -> str:
        parts = [f"{len(self.ok)} ok"]
        if self.drifted:
            parts.append(f"{len(self.drifted)} DRIFTED")
        if self.broke:
            parts.append(f"{len(self.broke)} BROKE")
        if self.known_drift:
            parts.append(f"{len(self.known_drift)} known-drift")
        if self.improved:
            parts.append(f"{len(self.improved)} improved")
        if self.still_error:
            parts.append(f"{len(self.still_error)} still unresolvable")
        return ", ".join(parts)


def _short_error(exc: BaseException) -> str:
    # Hydra errors carry multi-line "Available options" dumps; keep the first line only.
    text = str(exc).strip().split("\n")[0]
    return f"{type(exc).__name__}: {text}"[:300]


def _resolve(args: tuple[tp.Any, str, list[str]]) -> Entry:
    """Recompute the signature for one experiment. Module level so it pickles."""
    main, sig, argv = args
    try:
        got = main.get_xp(argv).sig
    except BaseException as exc:  # noqa: BLE001 - Hydra raises SystemExit for some errors
        return Entry(sig=sig, argv=argv, status=ERROR, error=_short_error(exc))
    return Entry(sig=sig, argv=argv, status=OK if got == sig else DRIFT, got_sig=got)


def iter_sigs(main) -> list[str]:
    """Every signature that has a cached argv, i.e. that Dora can still address."""
    xps = main.dora.dir / main.dora.xps
    if not xps.is_dir():
        return []
    return sorted(
        p.name for p in xps.iterdir() if SIG_RE.match(p.name) and (p / ".argv.json").exists()
    )


def build(main, sigs: tp.Sequence[str] | None = None, workers: int = 16) -> list[Entry]:
    """Recompute every signature, in parallel. Hydra keeps global state, so each
    worker gets its own process; `DecoratedMain` pickles by dotted name."""
    if sigs is None:
        sigs = iter_sigs(main)
    tasks = []
    for sig in sigs:
        try:
            argv = list(main.get_argv_from_sig(sig))
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            logging.getLogger(__name__).debug("Cannot load experiment metadata: %s", exc)
            continue
        tasks.append((main, sig, argv))
    if not tasks:
        return []
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_resolve, tasks, chunksize=4))


def check(main, corpus: list[Entry], workers: int = 16) -> Report:
    """Recompute the corpus and compare against its recorded baseline."""
    fresh = {e.sig: e for e in build(main, [e.sig for e in corpus], workers=workers)}
    report = Report()
    for old in corpus:
        new = fresh.get(old.sig)
        if new is None:
            report.broke.append((old.sig, "argv cache disappeared"))
            continue
        if new.status == OK:
            # Recomputing agrees with the folder name. Anything else in the
            # baseline means we just repaired it.
            if old.status != OK:
                report.improved.append(old.sig)
            report.ok.append(old.sig)
        elif new.status == DRIFT:
            if old.status == DRIFT and old.got_sig == new.got_sig:
                # Already drifted before this change, and drifted to the same
                # place: pre-existing damage, not a regression we introduced.
                report.known_drift.append(old.sig)
            else:
                report.drifted.append((old.sig, f"expected {old.sig}, got {new.got_sig}"))
        else:  # ERROR
            if old.status == ERROR:
                report.still_error.append(old.sig)
            else:
                report.broke.append((old.sig, new.error or "?"))
    return report


def save(path: Path, entries: list[Entry], main) -> None:
    payload = {
        "main": f"{main.package}.{main.main_module}",
        "dora_dir": str(main.dora.dir),
        "entries": [asdict(e) for e in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def load(path: Path) -> list[Entry]:
    payload = json.loads(Path(path).read_text())
    return [Entry(**e) for e in payload["entries"]]


def _main(argv: list[str] | None = None) -> int:
    from .._utils import get_main

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[0] not in ("build", "check"):
        print(__doc__)
        return 2
    action, path = args[0], Path(args[1])
    main = get_main()
    if action == "build":
        entries = build(main)
        save(path, entries, main)
        counts: dict[str, int] = {}
        for e in entries:
            counts[e.status] = counts.get(e.status, 0) + 1
        print(f"wrote {len(entries)} entries to {path}: {counts}")
        return 1 if counts.get(DRIFT) else 0
    report = check(main, load(path))
    print(report.summary())
    for sig, why in report.drifted + report.broke:
        print(f"  FAIL {sig}: {why}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(_main())
