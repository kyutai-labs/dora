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

None of these commands import the project when a `dora.toml` supplies the
experiment directory -- see `dora.project`.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import json
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


def note(message: str) -> None:
    """Diagnostics go to stderr; stdout stays machine-readable."""
    print(message, file=sys.stderr)


def emit(lines: tp.Sequence[str], as_json: tp.Any = None) -> None:
    """The single way anything here writes to stdout."""
    if as_json is not None:
        print(json.dumps(as_json, separators=(",", ":"), default=str))
        return
    text = "\n".join(lines)
    if len(text) > MAX_TOTAL_CHARS:
        text = (text[:MAX_TOTAL_CHARS]
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
    if not kept:
        return elide(parts[0], width)
    return " ".join(kept) + (f" +{dropped}" if dropped else "")


def columns(rows: tp.List[tp.List[str]], headers: tp.List[str]) -> tp.List[str]:
    """Fixed-width columns, no box drawing, no colour."""
    if not rows:
        return []
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
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
    resolution = resolve_targets(args.targets, dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    targets = resolution.targets
    if not targets:
        note("error: nothing to show")
        return 1

    limit = args.limit or MAX_ROWS
    shown = targets[:limit]

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

    if args.json:
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
            elide_parts(record["name"], MAX_NAME_CHARS),
            record["state"],
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
    lines += columns(rows, headers)
    if len(targets) > len(shown):
        lines.append(f"... {len(targets) - len(shown)} more (--limit)")
    tally: tp.Dict[str, int] = {}
    for record in records:
        tally[record["state"]] = tally.get(record["state"], 0) + 1
    lines.append(" | ".join(f"{state.lower()} {count}"
                            for state, count in sorted(tally.items())))
    emit(lines)
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

    if args.every:
        rows = rows[::args.every]
    limit = args.limit or MAX_METRIC_ROWS
    total = len(rows)
    if len(rows) > limit:
        rows = rows[-limit:]

    keys = ([k.strip() for k in args.keys.split(",")] if args.keys
            else choose_metric_keys([m for _, m in rows], MAX_METRIC_COLS))

    if args.json:
        emit([], as_json={
            "sig": target.sig, "stage": stage, "epochs": len(history),
            "rows": [{"epoch": e, **{k: m.get(k) for k in keys}} for e, m in rows]})
        return 0

    table = columns([[str(epoch)] + [fmt(metrics.get(k, "-")) for k in keys]
                     for epoch, metrics in rows], ["ep"] + keys)
    lines = [f"{target.sig} {stage}  {len(history)} epochs"
             + (f", showing {len(rows)} of {total}" if len(rows) < total else "")]
    lines += table
    emit(lines)
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


def clean(line: str) -> str:
    """Strip colour and cut the pathological lines.

    Hydra dumps every override onto a single line on error; on a real project
    that is ~4 KB, roughly a thousand tokens, for no benefit.
    """
    line = ANSI_RE.sub("", line.rstrip("\n"))
    if len(line) > MAX_LINE_CHARS:
        return line[:200] + f" ...[+{len(line) - 200} chars]"
    return line


def tail(path: Path, count: int) -> tp.List[str]:
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
    return [clean(ln) for ln in data.decode(errors="replace").splitlines()[-count:]]


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
    path = paths[0]
    count = min(args.limit or MAX_LOG_LINES, MAX_LOG_LINES_HARD)
    lines = tail(path, count if not args.grep else MAX_LOG_LINES_HARD)
    if args.grep:
        pattern = re.compile(args.grep)
        lines = [ln for ln in lines if pattern.search(ln)][-count:]
    if args.json:
        emit([], as_json={"sig": target.sig, "file": str(path), "lines": lines})
        return 0
    emit([f"{target.sig} {path.name} (last {len(lines)} lines)"] + lines)
    return 0


# Ordered by how specific the explanation is: a traceback says more than
# "the step died", which says more than a Slurm exit code.
FAILURE_PATTERNS: tp.List[tp.Tuple[str, str]] = [
    ("out of memory", r"torch\.OutOfMemoryError|CUDA out of memory|out of memory"),
    ("hydra config error", r"Error executing job with overrides|MissingConfigException"
                           r"|ConfigCompositionException"),
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


def extract_traceback(lines: tp.Sequence[str]) -> tp.List[str]:
    """The last Python traceback, compressed to its ends.

    Middle frames are almost always framework plumbing; the top says where it
    started and the bottom says what actually went wrong.
    """
    starts = [i for i, line in enumerate(lines)
              if line.lstrip().startswith("Traceback (most recent call last)")]
    if not starts:
        return []
    block = list(lines[starts[-1]:])
    if len(block) > 12:
        block = block[:4] + [f"  ... {len(block) - 10} frames ..."] + block[-6:]
    return block


def why_action(args: tp.Any, dora: DoraConfig) -> int:
    resolution = resolve_targets(args.targets[:1], dora)
    for problem in resolution.problems:
        note(f"warning: {problem}")
    if not resolution.targets:
        return 1
    target = resolution.targets[0]

    job = read_json(target.folder / "job.json") or {}
    job_id = job.get("job_id", "")
    state = job_states([job_id]).get(job_id, "?") if job_id else "-"

    paths = log_files(target, job_id=args.job)
    findings = []
    for path in paths[:16]:
        lines = tail(path, 400)
        label = classify(lines)
        trace = extract_traceback(lines)
        if label or trace:
            evidence = trace
            if not evidence and label:
                pattern = re.compile(
                    next(p for name, p in FAILURE_PATTERNS if name == label),
                    re.IGNORECASE)
                evidence = [ln for ln in lines if pattern.search(ln)][-3:]
            findings.append({"file": path.name, "cause": label, "traceback": evidence})

    if args.json:
        emit([], as_json={"sig": target.sig, "job": job_id, "state": state,
                          "findings": findings})
        return 0

    lines = [f"{target.sig}  job {job_id or '-'}  state {state}"]
    if not findings:
        # Nothing recognisable. The last words of the newest log are still the
        # best guess available, and are more use than admitting defeat.
        lines.append("no known failure signature; last lines of the newest log:")
        for path in paths[:1]:
            lines.append(f"[{path.name}]")
            lines += [ln for ln in tail(path, 12) if ln.strip()][-6:]
        if not paths:
            lines.append("(no logs at all for this experiment)")
        lines.append(f"more: dora log {target.sig} --tail 80")
    else:
        # Report one explanation, deduplicated: ranks fail together and the
        # same traceback on eight of them is one problem, not eight.
        seen: tp.Set[str] = set()
        for finding in findings:
            # Ranks fail together, and the same failure on eight of them is one
            # problem. Timestamps and rank numbers differ, so they are stripped
            # before comparing.
            body = "\n".join(finding["traceback"][-3:]) or str(finding["cause"])
            key = str(finding["cause"]) + re.sub(r"\d+", "#", body)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"[{finding['file']}] {finding['cause'] or 'traceback'}")
            lines += finding["traceback"]
            if len(seen) >= 2:
                break
        others = len(findings) - len(seen)
        if others > 0:
            lines.append(f"... same on {others} other rank(s)")
    emit(lines)
    return 0


def plan_action(args: tp.Any, main: tp.Any) -> int:
    """Resolve a grid to the experiments it would schedule, without scheduling.

    Unlike the read-only commands this has to import the project and evaluate
    the grid file -- there is no way to know what an explorer produces without
    running it -- so it is as slow as the project's import. What it avoids is
    the twenty kilobytes of table that `dora grid --dry_run` prints to say the
    same thing, and it names the experiments the grid has stopped producing,
    which a real launch would silently cancel.
    """
    from .conf import SubmitRules
    from .grid import RunGridArgs, _get_explore, run_grid

    explorer = _get_explore(args, main)
    grid_args = RunGridArgs(monitor=False, silent=True, dry_run=True,
                            patterns=list(args.patterns or []))
    sheeps = run_grid(main, explorer, args.grid, rules=SubmitRules(),
                      slurm=main.get_slurm_config(), args=grid_args)

    dora = main.dora
    grid_folder = dora.dir / dora._grids / args.grid
    produced = {sheep.xp.sig for sheep in sheeps}
    existing = ({c.name for c in grid_folder.iterdir()}
                if grid_folder.is_dir() else set())
    stale = sorted(existing - produced)

    try:
        names, base = main.get_names([sheep.xp for sheep in sheeps])
    except Exception:
        names, base = [sheep.xp.sig for sheep in sheeps], ""

    records = [{"index": i, "sig": sheep.xp.sig, "name": name or sheep.xp.sig,
                "launched": sheep.xp.sig in existing}
               for i, (sheep, name) in enumerate(zip(sheeps, names))]

    if args.json:
        emit([], as_json={"grid": args.grid, "experiments": records, "stale": stale})
        return 0

    limit = args.limit or MAX_ROWS
    rows = [[str(r["index"]), r["sig"], "launched" if r["launched"] else "NEW",
             elide_parts(r["name"], MAX_NAME_CHARS)] for r in records[:limit]]
    lines = [f"{args.grid}: {len(sheeps)} xps"]
    if base:
        lines.append("base: " + elide_parts(base, 160))
    lines += columns(rows, ["#", "sig", "status", "name"])
    if len(records) > limit:
        lines.append(f"... {len(records) - limit} more (--limit)")
    new = sum(1 for r in records if not r["launched"])
    lines.append(f"{len(records) - new} launched | {new} new")
    if stale:
        lines.append(f"WARNING: {len(stale)} experiment(s) in the grid folder are no "
                     f"longer produced and would be CANCELLED by a real launch: "
                     + " ".join(stale[:10]) + (" ..." if len(stale) > 10 else ""))
    emit(lines)
    return 0
