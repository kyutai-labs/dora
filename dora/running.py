"""Live running jobs, joined to persisted experiments and grid membership."""

import getpass
import subprocess as sp
from pathlib import Path
from typing import Any

from . import project
from .conf import DoraConfig
from .inspect import (
    JOB_RE,
    JSON,
    MAX_ROWS,
    PRETTY,
    Target,
    _xp_names,
    cap,
    columns,
    emit,
    format_name,
    get_name_width,
    note,
    output_mode,
    paint,
    read_json,
)


def gpu_count(tres: str) -> int | None:
    """Total allocated GPUs, without double counting typed GPU subtotals."""
    resources = dict(part.split("=", 1) for part in tres.split(",") if "=" in part)
    try:
        if "gres/gpu" in resources:
            return int(resources["gres/gpu"])
        typed = [int(v) for k, v in resources.items() if k.startswith("gres/gpu:")]
        if typed:
            return sum(typed)
        # A real allocation with no GPU entry is a CPU job; absent data is unknown.
        return 0 if resources else None
    except ValueError:
        return None


def running_jobs() -> list[dict[str, Any]]:
    """One scheduler query, including individual array tasks and hidden partitions."""
    proc = sp.run(
        [
            "squeue",
            "--local",
            "--all",
            "--noheader",
            "--array",
            "--states=RUNNING",
            "--user=" + getpass.getuser(),
            "--Format=JobArrayId:0|,Partition:0|,tres-alloc:0|,StdOut:0|,WorkDir:0|,Name:0",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    jobs = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 5)
        if len(parts) != 6 or not JOB_RE.fullmatch(parts[0].strip()):
            raise ValueError(f"Unexpected squeue output: {line!r}")
        job, partition, tres, stdout, workdir, name = (part.strip() for part in parts)
        jobs[job] = {
            "job": job,
            "partition": partition,
            "gpus": gpu_count(tres),
            "stdout": stdout,
            "workdir": workdir,
            "name": name,
        }
    return list(jobs.values())


def discover_projects(jobs: list[dict[str, Any]]) -> dict[str, DoraConfig]:
    """Find metadata from Slurm paths; never import code or evaluate a grid."""
    configs: dict[str, DoraConfig] = {}
    workdirs = {job.get("workdir", "") for job in jobs}
    for workdir in sorted(workdirs):
        if not Path(workdir).is_absolute():
            continue
        try:
            conf = project.load(Path(workdir))
            dora = conf.dora_config() if conf is not None else None
            if dora is not None:
                configs[str(dora.dir.resolve())] = dora
        except (OSError, RuntimeError, ValueError) as exc:
            note(f"warning: cannot read project settings at {workdir}: {exc}")
    # Single jobs log under ROOT/xps/SIG/submitit; arrays under ROOT/arrays/NAME.
    # These paths survive branch changes, deleted worktrees, and git-save clones.
    checked: set[Path] = set()
    for job in jobs:
        stdout = Path(job.get("stdout", ""))
        if not stdout.is_absolute():
            continue
        for root in stdout.parents:
            if root in checked:
                continue
            checked.add(root)
            try:
                if (root / "xps").is_dir() and (
                    (root / "by_id").is_dir() or (root / "grids").is_dir()
                ):
                    configs.setdefault(str(root.resolve()), DoraConfig(dir=root))
            except OSError:
                continue
    return configs


def _job_targets(dora: DoraConfig, jobs: list[dict[str, Any]]) -> dict[str, Target]:
    root = (dora.dir / dora.xps).resolve()
    targets = {}
    # by_id also retains earlier attempts still running after a newer submission.
    for job in jobs:
        folder = (dora.dir / dora.shep.by_id / job["job"]).resolve()
        if folder.parent == root and folder.is_dir():
            targets[job["job"]] = Target(folder.name, folder)
    # Dependents do not have by_id links. Never map array_job_ids to this XP:
    # those are siblings belonging to other experiments.
    live = {job["job"] for job in jobs}
    if live and root.is_dir():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.resolve().parent != root:
                continue
            info = read_json(folder / dora.shep.json_job_file)
            if not isinstance(info, dict):
                continue
            dependents = info.get("dependent_job_ids", [])
            ids = [info.get("job_id")]
            if isinstance(dependents, list):
                ids.extend(dependents)
            for job_id in ids:
                if isinstance(job_id, str) and job_id in live:
                    targets.setdefault(job_id, Target(folder.name, folder))
    return targets


def _memberships(dora: DoraConfig, targets: dict[str, Target]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {target.sig: [] for target in targets.values()}
    root = dora.dir / dora._grids
    folders = {target.folder.resolve(): target.sig for target in targets.values()}
    if root.is_dir():
        for grid in sorted(root.iterdir()):
            if grid.is_dir():
                for member in grid.iterdir():
                    sig = folders.get(member.resolve())
                    if sig is not None and grid.name not in memberships[sig]:
                        memberships[sig].append(grid.name)
    return memberships


def _totals(jobs: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "count": len(jobs),
        "gpus": sum(job["gpus"] for job in jobs if job["gpus"] is not None),
        "unknown_gpus": sum(job["gpus"] is None for job in jobs),
    }


def _gpu_label(totals: dict[str, Any]) -> str:
    suffix = f" + ? ({totals['unknown_gpus']} jobs)" if totals["unknown_gpus"] else ""
    return f"{totals['gpus']}{suffix}"


def running_action(args: Any, dora: DoraConfig | None = None) -> int:
    mode = output_mode(args)
    limit = (
        args.limit if args.limit is not None else (None if mode == JSON else cap(MAX_ROWS, mode))
    )
    if limit is not None and limit < 1:
        note("error: --limit must be positive")
        return 1
    try:
        live = running_jobs()
        configs = discover_projects(live)
        if dora is not None:
            configs[str(dora.dir.resolve())] = dora
        targets: dict[str, Target] = {}
        locations: dict[str, str] = {}
        memberships: dict[str, list[str]] = {}
        for config_root, config in configs.items():
            try:
                found = _job_targets(config, live)
                grids = _memberships(config, found)
            except OSError as exc:
                note(f"warning: cannot read experiments at {config_root}: {exc}")
                continue
            for job_id, found_target in found.items():
                targets[job_id] = found_target
                locations[job_id] = config_root
                memberships[job_id] = grids[found_target.sig]
    except (OSError, ValueError, sp.SubprocessError) as exc:
        note(f"error: cannot list running jobs: {exc}")
        if mode == JSON:
            emit([], as_json={"error": str(exc)})
        return 1

    jobs = []
    for job in live:
        target = targets.get(job["job"])
        jobs.append(
            {
                "job": job["job"],
                "partition": job["partition"],
                "gpus": job["gpus"],
                "sig": target.sig if target else None,
                "root": locations.get(job["job"]),
                "grids": memberships.get(job["job"], []),
                "name": job.get("name", job["job"]),
            }
        )
    jobs.sort(key=lambda job: (job["root"] or "", job["grids"], job["sig"] or "", job["job"]))
    shown = jobs if limit is None else jobs[:limit]
    shown_ids = {job["job"] for job in shown}
    grouped: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    for job in jobs:
        for grid in job["grids"] or [None]:
            grouped.setdefault((job["root"], grid), []).append(job)

    groups: list[dict[str, Any]] = []
    for (root, grid), members in sorted(
        grouped.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
    ):
        names: dict[str, str] = {}
        if root is not None:
            unique = {job["sig"]: targets[job["job"]] for job in members}
            names, _ = _xp_names(configs[root], list(unique.values()))
        records = [
            {**job, "name": names.get(job["sig"], job["name"])}
            for job in members
            if job["job"] in shown_ids
        ]
        groups.append(
            {
                "root": root,
                "grid": grid,
                **_totals(members),
                "shown": len(records),
                "jobs": records,
            }
        )
    result = {**_totals(jobs), "shown": len(shown), "grids": groups}
    if mode == JSON:
        emit([], as_json=result)
    else:
        name_width = get_name_width(dora) if mode == PRETTY else 100
        lines = [f"{len(jobs)} running job(s), {_gpu_label(result)} GPUs"]
        if any(len(job["grids"]) > 1 for job in jobs):
            lines.append("Shared XPs appear in each grid; overall totals count each job once.")
        for group in groups:
            label = group["grid"] or ("(no grid)" if group["root"] else "(unidentified)")
            if group["root"]:
                label += f" [{group['root']}]"
            lines += [
                "",
                paint(f"{label}: {group['count']} job(s), {_gpu_label(group)} GPUs", "1", mode),
            ]
            rows = [
                [
                    job["sig"] or "-",
                    job["job"],
                    str(job["gpus"]) if job["gpus"] is not None else "?",
                    job["partition"],
                    format_name(job["name"], mode, name_width),
                ]
                for job in group["jobs"]
            ]
            lines += [
                "  " + line
                for line in columns(rows, ["sig", "job", "GPUs", "partition", "name"], mode)
            ]
        if len(shown) < len(jobs):
            lines.append(f"... {len(jobs) - len(shown)} more jobs; totals include all jobs")
        emit(lines, mode=mode)
    return 0
