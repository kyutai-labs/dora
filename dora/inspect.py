# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Read-only commands for looking at experiments: `status`, `metrics`, `log`, `why`.

These exist because the existing output is shaped for a human watching a
terminal, and is expensive for anyone -- or anything -- reading it
programmatically. A single `dora grid --dry_run` on a twenty-experiment grid
prints 26 KB of ANSI-coloured, line-wrapped treetable; `dora info` prints an
experiment's whole argv twice. Both are fine to glance at and awful to consume.

So everything here goes through `emit`, which enforces the rules that make
output cheap to read: fixed-width columns, no colour, hard caps with an explicit
marker when something was cut, and `--json` for anything that wants to
post-process. Diagnostics go to stderr so stdout stays parseable.

Read-only invocations do not import the project when a `dora.toml` supplies the
experiment directory -- see `dora.project`.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess as sp
import sys
import typing as tp

from .conf import DoraConfig
from .names import NamesMixin
from .xp import XP, load_xp

# Hard caps. Everything is overridable with --limit, but the defaults are what
# make the output safe to read without checking its size first.
# Output modes. Pretty is for a person looking at a terminal: colour, and
# generous limits. Compact is for everything else -- a pipe, a file, a script,
# an agent -- where colour is noise and unbounded output is a hazard. The
# default is chosen by looking at stdout, the same way `ls` and `git` decide
# about colour, so neither audience has to remember a flag.
PRETTY = "pretty"
COMPACT = "compact"
JSON = "json"

# How much more a pretty rendering is allowed to show. Someone watching a
# terminal scrolls; a caller reading the output pays for every line.
PRETTY_SLACK = 5

MAX_ROWS = 40
MAX_METRIC_ROWS = 12
MAX_METRIC_COLS = 6
MAX_LOG_LINES = 40
MAX_LOG_LINES_HARD = 400
MAX_LINE_CHARS = 400
MAX_NAME_CHARS = 44
MAX_TOTAL_CHARS = 8000

SIG_RE = re.compile(r"^[0-9a-f]{8}$")
JOB_RE = re.compile(r"^\d+(_\d+)?$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def output_mode(args: tp.Any) -> str:
    """Pick the output mode: explicit flag first, otherwise by looking at stdout."""
    if getattr(args, "json", False):
        return JSON
    if getattr(args, "compact", False):
        return COMPACT
    if getattr(args, "pretty", False):
        return PRETTY
    try:
        interactive = sys.stdout.isatty()
    except (AttributeError, ValueError):
        interactive = False
    return PRETTY if interactive else COMPACT


def use_colour(mode: str) -> bool:
    """Colour only when pretty, and never if NO_COLOR is set (no-color.org)."""
    return mode == PRETTY and not os.environ.get("NO_COLOR")


def paint(text: str, colour: str, mode: str) -> str:
    if not use_colour(mode):
        return text
    from .log import colorize
    return colorize(text, colour)


# Slurm states worth telling apart at a glance.
_STATE_COLOURS = {
    "RUNNING": "32", "COMPLETED": "32", "PENDING": "33", "REQUEUED": "33",
    "FAILED": "31", "TIMEOUT": "31", "OUT_OF_MEMORY": "31", "NODE_FAIL": "31",
    "CANCELLED": "90", "MISSING": "90", "N/A": "90",
}


def paint_state(state: str, mode: str) -> str:
    colour = _STATE_COLOURS.get(state.upper())
    return paint(state, colour, mode) if colour else state


def cap(value: int, mode: str) -> int:
    """A limit, relaxed for a human reading the output."""
    return value * PRETTY_SLACK if mode == PRETTY else value


def note(message: str) -> None:
    """Diagnostics go to stderr; stdout stays machine-readable."""
    print(message, file=sys.stderr)


def emit(lines: tp.Sequence[str], as_json: tp.Any = None,
         mode: str = COMPACT) -> None:
    """The single way anything here writes to stdout."""
    if as_json is not None:
        print(json.dumps(as_json, separators=(",", ":"), default=str))
        return
    text = "\n".join(lines)
    limit = cap(MAX_TOTAL_CHARS, mode)
    if len(text) > limit:
        text = (text[:limit]
                + "\n... output truncated, narrow with --keys/--limit/--pattern")
    print(text)


def elide(text: str, width: int) -> str:
    """Shorten in the middle: for a path or a value, both ends carry meaning."""
    if len(text) <= width:
        return text
    keep = width - 3
    head = (keep + 1) // 2
    return text[:head] + "..." + text[len(text) - (keep - head):]


def elide_parts(name: str, width: int) -> str:
    """Shorten a name built of `key=value` parts, keeping whole parts.

    Cutting through the middle of such a name reliably destroys the bit that
    distinguishes one experiment from another, since the parts they share come
    first. Dropping whole parts from the end and saying how many at least leaves
    what is shown readable.
    """
    if len(name) <= width:
        return name
    parts = name.split(" ")
    kept: tp.List[str] = []
    used = 0
    for index, part in enumerate(parts):
        marker = f" +{len(parts) - index - 1}"
        if used + len(part) + len(marker) > width and kept:
            break
        kept.append(part)
        used += len(part) + 1
    dropped = len(parts) - len(kept)
    if len(kept) == 1 and len(kept[0]) > width:
        # A single part longer than the whole budget: nothing to drop, so fall
        # back to cutting through it rather than blowing the column open.
        head = elide(kept[0], width - (len(f" +{dropped}") if dropped else 0))
        return head + (f" +{dropped}" if dropped else "")
    return " ".join(kept) + (f" +{dropped}" if dropped else "")


def _visible_len(text: str) -> int:
    """Length as displayed, ignoring colour codes, so columns still line up."""
    return len(ANSI_RE.sub("", text))


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _visible_len(text))


def columns(rows: tp.List[tp.List[str]], headers: tp.List[str],
            mode: str = COMPACT) -> tp.List[str]:
    """Fixed-width columns, no box drawing. Colour only in pretty mode."""
    if not rows:
        return []
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(cell))
    header = "  ".join(_pad(h, widths[i]) for i, h in enumerate(headers)).rstrip()
    out = [paint(header, "1", mode)]
    for row in rows:
        out.append("  ".join(_pad(cell, widths[i]) for i, cell in enumerate(row)).rstrip())
    return out


def fmt(value: tp.Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def ago(seconds: float) -> str:
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds / size:.1f}{unit}"
    return f"{seconds:.0f}s"


class _DeltaNamer(NamesMixin):
    """Names experiments without importing the project.

    Uses the persisted delta when there is one. Failing that it falls back to
    parsing the argv, which is approximate -- it keeps overrides that matched the
    default and misses the subtree collapsing the real delta does -- but a name
    is only ever for a human to read, and an approximate name beats a bare
    signature. Signatures are never computed from this.
    """

    def __init__(self, dora: tp.Optional[DoraConfig] = None):
        self.dora = dora

    def get_name_parts(self, xp: XP) -> OrderedDict:
        if xp.delta is not None:
            return OrderedDict(xp.delta)
        parts: OrderedDict = OrderedDict()
        for arg in xp.argv:
            if "=" not in arg:
                continue
            key, value = arg.split("=", 1)
            key = key.lstrip("+~")
            if self.dora is not None and self.dora.is_excluded(key):
                continue
            parts[key] = value.strip('"')
        return parts


@dataclass
class Target:
    sig: str
    folder: Path
    grid: tp.Optional[str] = None


@dataclass
class Resolution:
    targets: tp.List[Target] = field(default_factory=list)
    problems: tp.List[str] = field(default_factory=list)


def resolve_targets(tokens: tp.Sequence[str], dora: DoraConfig,
                    grid_module_exists: tp.Optional[tp.Callable[[str], bool]] = None
                    ) -> Resolution:
    """Turn user-supplied tokens into experiments.

    A token may be a signature, a grid name, a Slurm job id, or `@sig` to force
    signature interpretation for a folder that no longer looks like one.
    """
    out = Resolution()
    xps_root = dora.dir / dora.xps
    grids_root = dora.dir / dora._grids
    seen: tp.Set[str] = set()

    def add(sig: str, grid: tp.Optional[str] = None) -> None:
        if sig in seen:
            return
        seen.add(sig)
        out.targets.append(Target(sig=sig, folder=xps_root / sig, grid=grid))

    for token in tokens:
        if token.startswith("@"):
            add(token[1:])
            continue
        # Checked before job ids: a signature is eight hex characters, which
        # can be all digits, and would otherwise be mistaken for a job id.
        if (xps_root / token).is_dir():
            add(token)
            continue
        if JOB_RE.match(token):
            link = dora.dir / dora.shep.by_id / token
            if link.exists():
                add(link.resolve().name)
            else:
                out.problems.append(f"no experiment for job id {token}")
            continue
        grid_folder = grids_root / token
        if grid_folder.is_dir():
            children = [c for c in grid_folder.iterdir() if c.is_symlink() or c.is_dir()]
            # Order by when they joined the grid, so indices are stable.
            for child in sorted(children, key=lambda c: (c.lstat().st_ctime, c.name)):
                add(child.name, grid=token)
            if not children:
                out.problems.append(f"grid {token} has no experiments")
            continue
        if SIG_RE.match(token):
            out.problems.append(f"no experiment folder for signature {token}")
            continue
        # A grid that exists as a file but was never launched, or a typo.
        if grid_module_exists is not None and grid_module_exists(token):
            out.problems.append(
                f"grid {token} has never been launched (no experiments recorded)")
        else:
            hint = _closest(token, grids_root)
            out.problems.append(
                f"no signature, grid or job id {token!r}"
                + (f"; did you mean {hint}?" if hint else ""))
    return out


def _closest(token: str, grids_root: Path, limit: int = 3) -> str:
    if not grids_root.is_dir():
        return ""
    import difflib
    names = [p.name for p in grids_root.iterdir()]
    return ", ".join(difflib.get_close_matches(token, names, n=limit, cutoff=0.5))


def read_json(path: Path, retries: int = 1) -> tp.Any:
    """Read a JSON file that a running job may be rewriting underneath us."""
    import time
    for attempt in range(retries + 1):
        try:
            with open(path) as fileobj:
                return json.load(fileobj)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, ValueError):
            if attempt == retries:
                return None
            time.sleep(0.05)
    return None


def job_states(job_ids: tp.Sequence[str]) -> tp.Dict[str, str]:
    """Ask Slurm about every job at once.

    `-X` collapses the `.batch`/`.extern` steps, which would otherwise return
    several rows per job.
    """
    ids = [j for j in job_ids if j]
    if not ids:
        return {}
    try:
        proc = sp.run(["sacct", "-X", "--parsable2", "-n", "-o", "JobID,State",
                       "-j", ",".join(ids)],
                      capture_output=True, check=True, timeout=30)
    except (OSError, sp.SubprocessError):
        return {}
    states = {}
    for line in proc.stdout.decode(errors="replace").strip().splitlines():
        if "|" not in line:
            continue
        job_id, state = line.split("|", 1)
        # "CANCELLED by 2012" and friends carry a trailing explanation.
        states[job_id.strip()] = state.strip().split()[0] if state.strip() else "?"
    return states


def last_activity(folder: Path) -> tp.Optional[float]:
    """When this experiment last wrote anything, as a Unix timestamp."""
    newest = None
    for pattern in ("solver.log.*", "history.json", "train.log"):
        for path in folder.glob(pattern):
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            if newest is None or stamp > newest:
                newest = stamp
    return newest


# ---------------------------------------------------------------- history


def flatten_history(history: tp.List[dict], stage: tp.Optional[str] = None
                    ) -> tp.List[tp.Tuple[int, dict]]:
    """`[(epoch, metrics)]` for one stage, epochs 1-based as Dora counts them."""
    out = []
    for index, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            continue
        if stage is None:
            merged: dict = {}
            for name, metrics in entry.items():
                if isinstance(metrics, dict):
                    merged.update({f"{name}.{k}": v for k, v in metrics.items()})
            if merged:
                out.append((index, merged))
        elif isinstance(entry.get(stage), dict):
            out.append((index, entry[stage]))
    return out


def pick_stage(history: tp.List[dict]) -> tp.Optional[str]:
    """Prefer the stage a person would look at first."""
    present = {name for entry in history if isinstance(entry, dict) for name in entry}
    for candidate in ("valid", "train", "evaluate"):
        if candidate in present:
            return candidate
    return sorted(present)[0] if present else None


def summarise(folder: Path, keys: tp.Sequence[str]) -> tp.Tuple[int, dict]:
    """Epoch count and last metrics, without loading more than needed."""
    history = read_json(folder / "history.json")
    if not isinstance(history, list) or not history:
        return 0, {}
    stage = pick_stage(history)
    rows = flatten_history(history, stage)
    if not rows:
        return len(history), {}
    _, last = rows[-1]
    if keys:
        last = {k: v for k, v in last.items() if k in keys}
    return len(history), last


def choose_metric_keys(samples: tp.Sequence[dict], limit: int) -> tp.List[str]:
    """Pick the few metrics worth a column.

    Preference for the names people actually track, then whatever is left, so a
    project with unusual metric names still shows something useful.
    """
    preferred = ("loss", "ce", "ppl", "nll", "acc", "wer", "reward", "grad_norm")
    seen: tp.List[str] = []
    for sample in samples:
        for key in sample:
            if key not in seen:
                seen.append(key)
    ranked = sorted(seen, key=lambda k: (
        min((i for i, p in enumerate(preferred) if p in k.lower()), default=len(preferred)),
        len(k)))
    return ranked[:limit]


# ---------------------------------------------------------------- actions


def _xp_names(dora: DoraConfig,
              targets: tp.List[Target]) -> tp.Tuple[tp.Dict[str, str], str]:
    """Short names for a set of experiments, plus the part they all share.

    Factoring the common overrides into a single header line is what keeps the
    per-experiment names down to the bit that actually differs.
    """
    xps, sigs = [], []
    for target in targets:
        try:
            xps.append(load_xp(dora, target.sig))
            sigs.append(target.sig)
        except Exception:
            continue
    if not xps:
        return {}, ""
    try:
        names, base = _DeltaNamer(dora).get_names(xps)
    except Exception:
        return {sig: sig for sig in sigs}, ""
    return {sig: (name or sig) for sig, name in zip(sigs, names)}, base


def status_action(args: tp.Any, dora: DoraConfig) -> int:
    if getattr(args, "cancel", False) or getattr(args, "restart", False):
        from .manage import status_action as manage_status
        return manage_status(args, dora)
    resolution = resolve_targets(args.targets, dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    targets = resolution.targets
    if not targets:
        note("error: nothing to show")
        return 1

    mode = output_mode(args)
    # JSON is the format a script asks for when it wants the data, so silently
    # dropping rows would be the wrong kindness; it is complete unless --limit
    # says otherwise. The rendered forms stay capped.
    limit = args.limit or (None if mode == JSON else cap(MAX_ROWS, mode))
    shown = targets if limit is None else targets[:limit]

    jobs = {}
    for target in shown:
        job = read_json(target.folder / "job.json")
        jobs[target.sig] = (job or {}).get("job_id", "")
    states = job_states(list(jobs.values()))
    names, base_name = _xp_names(dora, shown)
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else []

    records = []
    for index, target in enumerate(shown):
        epoch, metrics = summarise(target.folder, keys)
        stamp = last_activity(target.folder)
        job_id = jobs[target.sig]
        records.append({
            "index": index,
            "sig": target.sig,
            "name": names.get(target.sig, target.sig),
            "state": states.get(job_id, "-" if not job_id else "?"),
            "job": job_id,
            "epoch": epoch,
            "ping": None if stamp is None else round(_now() - stamp, 1),
            "metrics": metrics,
        })

    if mode == JSON:
        emit([], as_json={"count": len(targets), "shown": len(shown),
                          "experiments": records})
        return 0

    metric_keys = keys or choose_metric_keys([r["metrics"] for r in records],
                                             4 if not keys else MAX_METRIC_COLS)
    headers = ["#", "sig", "name", "state", "job", "ep", "ping"] + metric_keys
    rows = []
    for record in records:
        rows.append([
            str(record["index"]),
            record["sig"],
            elide_parts(record["name"], cap(MAX_NAME_CHARS, mode)),
            paint_state(record["state"], mode),
            record["job"] or "-",
            str(record["epoch"]),
            "-" if record["ping"] is None else ago(record["ping"]),
        ] + [fmt(record["metrics"].get(k, "-")) for k in metric_keys])

    lines = []
    grids = {t.grid for t in shown if t.grid}
    if len(grids) == 1:
        lines.append(f"grid {grids.pop()} ({len(targets)} xps)")
    if base_name:
        lines.append("base: " + elide_parts(base_name, 160))
    lines += columns(rows, headers, mode)
    if len(targets) > len(shown):
        lines.append(f"... {len(targets) - len(shown)} more (--limit)")
    tally: tp.Dict[str, int] = {}
    for record in records:
        tally[record["state"]] = tally.get(record["state"], 0) + 1
    lines.append(" | ".join(f"{state.lower()} {count}"
                            for state, count in sorted(tally.items())))
    emit(lines, mode=mode)
    return 0


def _now() -> float:
    import time
    return time.time()


def metrics_action(args: tp.Any, dora: DoraConfig) -> int:
    resolution = resolve_targets(args.targets[:1], dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    if not resolution.targets:
        return 1
    target = resolution.targets[0]
    history = read_json(target.folder / "history.json")
    if not isinstance(history, list) or not history:
        note(f"error: no history for {target.sig}")
        return 1

    stage = args.stage or pick_stage(history)
    rows = flatten_history(history, stage)
    if not rows:
        note(f"error: no stage {stage!r} in {target.sig}")
        return 1

    mode = output_mode(args)
    if args.every:
        rows = rows[::args.every]
    limit = args.limit or cap(MAX_METRIC_ROWS, mode)
    total = len(rows)
    if len(rows) > limit:
        rows = rows[-limit:]

    keys = ([k.strip() for k in args.keys.split(",")] if args.keys
            else choose_metric_keys([m for _, m in rows], cap(MAX_METRIC_COLS, mode)))

    if mode == JSON:
        emit([], as_json={
            "sig": target.sig, "stage": stage, "epochs": len(history),
            "rows": [{"epoch": e, **{k: m.get(k) for k in keys}} for e, m in rows]})
        return 0

    table = columns([[str(epoch)] + [fmt(metrics.get(k, "-")) for k in keys]
                     for epoch, metrics in rows], ["ep"] + keys, mode)
    lines = [f"{target.sig} {stage}  {len(history)} epochs"
             + (f", showing {len(rows)} of {total}" if len(rows) < total else "")]
    lines += table
    emit(lines, mode=mode)
    return 0


# ---------------------------------------------------------------- logs


def log_files(target: Target, rank: tp.Optional[int] = None,
              job_id: tp.Optional[str] = None) -> tp.List[Path]:
    """Every log for an experiment, most recently written first.

    Dora runs submitit with `stderr_to_stdout`, so there are no `.err` files;
    everything is in `<job>_<task>_log.out`. The `latest` symlink points at the
    array folder for array members, where the job id itself looks like `12_3`,
    so the files are found by globbing rather than by formatting a name.
    """
    found: tp.List[Path] = []
    submitit = target.folder / "latest"
    if not submitit.exists():
        submitit = target.folder / "submitit"
    if submitit.exists():
        pattern = f"*{job_id}*_log.out" if job_id else "*_log.out"
        found += sorted(submitit.glob(pattern))
    if rank is None:
        found += sorted(target.folder.glob("solver.log.*"))
    else:
        found += sorted(target.folder.glob(f"solver.log.{rank}"))
    found += sorted(target.folder.glob("train.log"))
    return sorted({p for p in found if p.is_file()},
                  key=lambda p: p.stat().st_mtime, reverse=True)


LOG_NAME_RE = re.compile(r"^(?P<job>\d+(?:_\d+)?)_(?P<task>\d+)_log\.out$")


def log_attempts(target: Target,
                 current_job: tp.Optional[str] = None
                 ) -> tp.List[tp.Tuple[str, tp.List[Path]]]:
    """Group an experiment's logs by the job attempt that produced them.

    An experiment is usually run more than once -- requeued, resubmitted after a
    fix, restarted from a checkpoint -- and each attempt writes a full set of
    per-rank logs. Sorting all of them by modification time interleaves the
    attempts and buries an older failure under a newer success, so "why did this
    fail" has to be asked per attempt, newest first.

    `current_job` -- from the experiment's `job.json` -- is put first, because
    it is the only authoritative statement of which attempt is current. Note
    that a larger job id does not mean a later job: Slurm's accounting database
    gets reset and ids start again from a low number. Everything else falls back
    to modification time, which stays correct across such a reset.

    Returns `[(attempt, paths)]`, newest attempt first. `solver.log.*` are the
    current run's per-rank logs and are grouped under "solver".
    """
    groups: tp.Dict[str, tp.List[Path]] = {}
    submitit = target.folder / "latest"
    if not submitit.exists():
        submitit = target.folder / "submitit"
    if submitit.exists():
        for path in submitit.glob("*_log.out"):
            match = LOG_NAME_RE.match(path.name)
            groups.setdefault(match.group("job") if match else "?", []).append(path)
    solver = sorted(target.folder.glob("solver.log.*"))
    if solver:
        groups["solver"] = solver
    train = target.folder / "train.log"
    if train.is_file():
        groups.setdefault("train", []).append(train)

    def newest(paths: tp.List[Path]) -> float:
        return max((p.stat().st_mtime for p in paths), default=0.0)

    ordered = sorted(groups.items(), key=lambda kv: newest(kv[1]), reverse=True)
    if current_job:
        ordered.sort(key=lambda kv: kv[0] != current_job)
    return ordered


def clean(line: str, mode: str = COMPACT) -> str:
    """Trim a log line for display.

    Solver logs are colourised, which is exactly what a person reading them in
    a terminal wants and exactly what anything else does not, so the escapes
    are kept in pretty mode and stripped otherwise. Long lines are cut either
    way: Hydra dumps every override onto one line on error, which on a real
    project is ~4KB, roughly a thousand tokens, for no benefit.
    """
    line = line.rstrip("\n")
    if not use_colour(mode):
        line = ANSI_RE.sub("", line)
    limit = cap(MAX_LINE_CHARS, mode)
    if _visible_len(line) > limit:
        return ANSI_RE.sub("", line)[:limit // 2] + \
            f" ...[+{_visible_len(line) - limit // 2} chars]"
    return line


def tail(path: Path, count: int, mode: str = COMPACT) -> tp.List[str]:
    """Last `count` lines, read from the end rather than through the file."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    block, data = 65536, b""
    with open(path, "rb") as fileobj:
        while size > 0 and data.count(b"\n") <= count:
            step = min(block, size)
            size -= step
            fileobj.seek(size)
            data = fileobj.read(step) + data
    return [clean(ln, mode)
            for ln in data.decode(errors="replace").splitlines()[-count:]]


def log_action(args: tp.Any, dora: DoraConfig) -> int:
    resolution = resolve_targets(args.targets[:1], dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    if not resolution.targets:
        return 1
    target = resolution.targets[0]
    paths = log_files(target, rank=args.rank, job_id=args.job)
    if not paths:
        note(f"error: no logs for {target.sig}")
        return 1
    mode = output_mode(args)
    path = paths[0]
    hard = cap(MAX_LOG_LINES_HARD, mode)
    count = min(args.limit or cap(MAX_LOG_LINES, mode), hard)
    lines = tail(path, count if not args.grep else hard, mode)
    if args.grep:
        pattern = re.compile(args.grep)
        lines = [ln for ln in lines if pattern.search(ANSI_RE.sub("", ln))][-count:]
    if mode == JSON:
        # JSON never carries escape codes, whatever the terminal is doing.
        emit([], as_json={"sig": target.sig, "file": str(path),
                          "lines": [ANSI_RE.sub("", ln) for ln in lines]})
        return 0
    header = f"{target.sig} {path.name} (last {len(lines)} lines)"
    emit([paint(header, "1", mode)] + lines, mode=mode)
    return 0


# Ordered by how specific the explanation is: a traceback says more than
# "the step died", which says more than a Slurm exit code.
FAILURE_PATTERNS: tp.List[tp.Tuple[str, str]] = [
    ("out of memory", r"torch\.OutOfMemoryError|CUDA out of memory|out of memory"),
    ("hydra config error", r"MissingConfigException|ConfigCompositionException"),
    ("NCCL / collective timeout", r"NCCL.*(timeout|error)|Watchdog caught|ProcessGroupNCCL"),
    ("could not start the job", r"execve\(\)|command not found|No such file or directory"),
    ("killed by Slurm", r"DUE TO TIME LIMIT|CANCELLED AT|oom-kill|slurmstepd: error"
                        r"|srun: error"),
]


def classify(lines: tp.Sequence[str]) -> tp.Optional[str]:
    for label, pattern in FAILURE_PATTERNS:
        expr = re.compile(pattern, re.IGNORECASE)
        if any(expr.search(line) for line in lines):
            return label
    return None


# Lines that continue a traceback rather than ending it.
_CHAINED = ("During handling of the above exception",
            "The above exception was the direct cause",
            "Traceback (most recent call last)")


def extract_traceback(lines: tp.Sequence[str]) -> tp.List[str]:
    """The last Python traceback, bounded and compressed to its ends.

    Bounding matters more than it sounds. A traceback is followed by whatever
    the process printed on its way down -- for a distributed job that is
    hundreds of lines of NCCL teardown -- so running to the end of the file
    buries the exception, which is the one line anybody wanted. A traceback ends
    at the first unindented line after the header, and that line *is* the
    exception, unless it says the exception was chained, in which case the real
    one is further down.

    Middle frames are almost always framework plumbing, so the top (where it
    started) and the bottom (where it broke) are kept and the rest counted.
    """
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("Traceback (most recent call last)")]
    if not starts:
        return []

    block = [lines[starts[-1]]]
    for line in lines[starts[-1] + 1:]:
        if not line.strip() or line.startswith((" ", "\t")):
            block.append(line)
            continue
        block.append(line)
        if not line.startswith(_CHAINED):
            break  # the exception line: the traceback ends here
    while block and not block[-1].strip():
        block.pop()

    if len(block) > 12:
        head, tail_ = block[:4], block[-5:]
        block = head + [f"  ... {len(block) - len(head) - len(tail_)} frames ..."] + tail_
    return block


def exception_line(trace: tp.Sequence[str]) -> tp.Optional[str]:
    """The `SomeError: message` a traceback ends on, if it looks like one."""
    if not trace:
        return None
    last = trace[-1].strip()
    if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning)\b", last):
        return last
    return last if ":" in last and not last.startswith(("File ", "  ")) else None


def why_action(args: tp.Any, dora: DoraConfig) -> int:
    resolution = resolve_targets(args.targets[:1], dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    if not resolution.targets:
        return 1
    target = resolution.targets[0]

    mode = output_mode(args)
    job = read_json(target.folder / "job.json") or {}
    current_job = job.get("job_id", "")

    attempts = log_attempts(target, current_job=current_job)
    if args.job:
        attempts = [(name, paths) for name, paths in attempts if name == args.job]

    states = job_states([name for name, _ in attempts if name.isdigit()]
                        + ([current_job] if current_job else []))
    current_state = states.get(current_job, "?") if current_job else "-"

    # Walk attempts newest first and stop at the first one that explains
    # something. An older failure is still the answer when the newest attempt
    # merely ran out of time or is still going.
    culprit: tp.Optional[str] = None
    findings: tp.List[dict] = []
    scanned = 0
    for name, paths in attempts:
        if scanned >= args.attempts:
            break
        scanned += 1
        for path in sorted(paths):
            lines = tail(path, 1500)
            trace = extract_traceback(lines)
            label = classify(lines)
            if trace:
                # The exception names the failure far better than any pattern.
                # Hydra wraps every exception in main with "Error executing job
                # with overrides", so trusting patterns here would report a
                # missing data file as a configuration error.
                label = exception_line(trace) or label
            if not (label or trace):
                continue
            evidence = trace
            if not evidence and label:
                patterns = [p for lab, p in FAILURE_PATTERNS if lab == label]
                if patterns:
                    expr = re.compile(patterns[0], re.IGNORECASE)
                    evidence = [ln for ln in lines if expr.search(ln)][-3:]
            findings.append({"attempt": name, "file": path.name,
                             "cause": label, "traceback": evidence})
        if findings:
            culprit = name
            break

    if mode == JSON:
        emit([], as_json={"sig": target.sig, "job": current_job,
                          "state": current_state, "attempt": culprit,
                          "attempts": [name for name, _ in attempts],
                          "findings": findings})
        return 0

    header = (f"{target.sig}  job {current_job or '-'}  state "
              + paint_state(current_state, mode))
    jobs = [name for name, _ in attempts if name.isdigit()]
    if len(jobs) > 1:
        header += f"  ({len(jobs)} attempts: " + " ".join(jobs[:6]) + ")"
    lines = [header]

    if not findings:
        lines.append("no known failure signature; last lines of the newest log:")
        for _, paths in attempts[:1]:
            path = sorted(paths)[0]
            lines.append(f"[{path.name}]")
            lines += [ln for ln in tail(path, 12) if ln.strip()][-6:]
        if not attempts:
            lines.append("(no logs at all for this experiment)")
        lines.append(f"more: dora log {target.sig} --tail 80")
        emit(lines, mode=mode)
        return 0

    attempt_state = states.get(culprit or "", "")
    if culprit and culprit != current_job and culprit.isdigit():
        lines.append(f"failure is from attempt {culprit}"
                     + (f" ({attempt_state})" if attempt_state else "")
                     + f", not the current job {current_job or '-'}")
    # Ranks fail together, so report the cause once and count the rest.
    seen: tp.Set[str] = set()
    for finding in findings:
        body = "\n".join(finding["traceback"][-3:]) or str(finding["cause"])
        key = str(finding["cause"]) + re.sub(r"\d+", "#", body)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{finding['file']}] "
                     + paint(str(finding["cause"] or "traceback"), "31", mode))
        lines += finding["traceback"]
        if len(seen) >= 2:
            break
    others = len(findings) - len(seen)
    if others > 0:
        lines.append(f"... same on {others} other rank(s)")
    emit(lines, mode=mode)
    return 0


def render_grid(args: tp.Any, herd: tp.Sequence[tp.Any], names: tp.Sequence[str],
                base_name: str, lines: tp.Sequence[dict],
                stale: tp.Sequence[tp.Any] = ()) -> None:
    """Compact rendering of a grid, for `dora grid --compact` / `--json`.

    The treetable this replaces is built for a human watching a terminal: it is
    coloured, it wraps, and on a twenty-experiment grid it runs to 26KB. This
    says the same thing in a tenth of that, and stays readable when something
    else has to parse it.
    """
    records = []
    for index, (sheep, name, line) in enumerate(zip(herd, names, lines)):
        meta = line.get("Meta", {})
        metrics: tp.Dict[str, tp.Any] = {}
        for group, values in line.items():
            if group == "Meta" or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[f"{group}.{key}" if group else key] = value
        records.append({
            "index": index,
            "sig": sheep.xp.sig,
            "name": name or sheep.xp.sig,
            "state": meta.get("state", "N/A"),
            "job": meta.get("sid", "") or "",
            "metrics": metrics,
        })

    # A stale experiment that is still running gets cancelled; one that already
    # finished is only dropped from the grid and keeps its results. Conflating
    # the two would make a harmless edit look alarming.
    live_stale, done_stale = [], []
    for sheep in stale:
        (done_stale if sheep.is_done() else live_stale).append(sheep.xp.sig)

    mode = output_mode(args)
    if mode == JSON:
        wanted = getattr(args, "limit", None)
        emit([], as_json={"experiments": records[:wanted] if wanted else records,
                          "base_name": base_name,
                          "would_cancel": live_stale, "would_drop": done_stale})
        return

    limit = getattr(args, "limit", None) or cap(MAX_ROWS, mode)
    shown = records[:limit]  # only the rendered form is capped; JSON left above
    metric_keys = choose_metric_keys([r["metrics"] for r in shown], cap(4, mode))
    rows = [[str(r["index"]), r["sig"],
             elide_parts(r["name"], cap(MAX_NAME_CHARS, mode)),
             paint_state(r["state"], mode), r["job"] or "-"]
            + [fmt(r["metrics"].get(k, "-")) for k in metric_keys]
            for r in shown]

    out = []
    if base_name:
        out.append("base: " + elide_parts(base_name, 160))
    out += columns(rows, ["#", "sig", "name", "state", "job"] + metric_keys, mode)
    if len(records) > len(shown):
        out.append(f"... {len(records) - len(shown)} more (--limit)")
    tally: tp.Dict[str, int] = {}
    for record in records:
        tally[record["state"]] = tally.get(record["state"], 0) + 1
    out.append(" | ".join(f"{state.lower()} {count}"
                          for state, count in sorted(tally.items())))
    if live_stale:
        out.append(paint("WARNING:", "31", mode)
                   + f" {len(live_stale)} running experiment(s) would be CANCELLED, "
                   "the grid no longer produces them: " + " ".join(live_stale[:10])
                   + (" ..." if len(live_stale) > 10 else ""))
    if done_stale:
        out.append(f"{len(done_stale)} finished experiment(s) would be dropped from the "
                   "grid (results kept): " + " ".join(done_stale[:10])
                   + (" ..." if len(done_stale) > 10 else ""))
    emit(out, mode=mode)
