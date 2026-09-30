"""Explicit actions on persisted experiments, without evaluating a grid."""

import json
from pathlib import Path
import subprocess as sp
from typing import Any

from .conf import DoraConfig, SlurmConfig, SubmitRules
from .inspect import (
    JOB_RE,
    MAX_ROWS,
    cap,
    emit,
    job_states,
    note,
    output_mode,
    resolve_targets,
    JSON,
)
from .shep import Sheep, Shepherd

_TERMINAL = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "OUT_OF_MEMORY",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
    "MISSING",
}


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def load_slurm_config(folder: Path, dora: DoraConfig) -> SlurmConfig:
    """Use slurm.json, falling back to the snapshot in older job.json files."""
    path = folder / "slurm.json"
    if path.exists():
        raw = _read(path)
    else:
        path = folder / dora.shep.json_job_file
        job = _read(path)
        raw = job.get("slurm_config") if isinstance(job, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: no saved Slurm parameters; launch this XP from its grid first")
    try:
        return SlurmConfig(**raw)
    except TypeError as exc:
        raise ValueError(f"{path}: invalid Slurm parameters: {exc}") from exc


def _job_ids(folder: Path, dora: DoraConfig) -> list[str]:
    path = folder / dora.shep.json_job_file
    if not path.exists():
        return []
    job = _read(path)
    if not isinstance(job, dict):
        raise ValueError(f"{path}: expected job metadata")
    first = job.get("job_id")
    dependents = job.get("dependent_job_ids", [])
    if not isinstance(dependents, list):
        raise ValueError(f"{path}: expected a list of dependent job ids")
    ids = ([first] if first is not None else []) + dependents
    if any(not isinstance(value, str) or not JOB_RE.fullmatch(value) for value in ids):
        raise ValueError(f"{path}: invalid Slurm job id")
    # Array siblings belong to other experiments and must not be cancelled.
    return list(dict.fromkeys(ids))


def status_action(args, dora: DoraConfig) -> int:
    """Cancel or restart all explicitly selected XPs; --limit only limits output."""
    action = "restart" if args.restart else "cancel"
    dry_run = args.dry_run
    resolution = resolve_targets(args.targets, dora)
    records = []
    try:
        if resolution.problems:
            raise ValueError("; ".join(resolution.problems))
        if not resolution.targets:
            raise ValueError("No experiments selected")
        plans = []
        root = (dora.dir / dora.xps).resolve()
        for target in resolution.targets:
            if target.folder.resolve().parent != root or not target.folder.is_dir():
                raise ValueError(f"{target.sig}: experiment does not exist inside {root}")
            ids = _job_ids(target.folder, dora)
            slurm = load_slurm_config(target.folder, dora) if args.restart else None
            if slurm is not None and getattr(args, "partition", None) is not None:
                slurm.partition = args.partition
            plans.append((target, ids, slurm))

        shepherd = None
        sheeps = []
        if args.restart:
            # Import the entry point once. Never import or evaluate a grid.
            from ._utils import get_main

            main = get_main(args.main_module, args.package)
            for target, ids, slurm in plans:
                argv = _read(target.folder / ".argv.json")
                if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
                    raise ValueError(f"{target.sig}: invalid saved experiment arguments")
                xp = main.get_xp(argv)
                if xp.sig != target.sig or xp.folder.resolve() != target.folder.resolve():
                    raise ValueError(
                        f"{target.sig}: current project/config produces {xp.sig} at {xp.folder}; "
                        "refusing to restart a different experiment"
                    )
                sheep = Sheep(xp)
                sheep.job = None
                sheep._other_jobs = []
                sheep._dependent_jobs = []
                sheeps.append(sheep)

            if main.dora.git_save:
                from . import git_save

                if dry_run:
                    git_save.check_repo_clean(git_save.get_git_root(), main)
                else:
                    # Clone and run setup hooks before stopping a working job.
                    git_save.get_new_clone(main)

        # Complete preflight for every target before cancelling or submitting.
        ids = list(dict.fromkeys(job for _, jobs, _ in plans for job in jobs))
        states = job_states(ids)
        to_cancel = [job for job in ids if states.get(job) not in _TERMINAL]
        if not dry_run:
            if args.restart:
                # Targeted actions must not run the unrelated orphan-job recovery.
                shepherd = Shepherd(main, check_orphans=False)
            if to_cancel:
                sp.run(["scancel", *to_cancel], check=True, capture_output=True, text=True)

        for index, (target, ids, slurm) in enumerate(plans):
            record: dict[str, Any] = {
                "sig": target.sig,
                "action": action,
                "cancel_jobs": [job for job in ids if job in to_cancel],
            }
            if args.restart:
                assert slurm is not None
                record["partition"] = slurm.partition
                sheep = sheeps[index]
                if shepherd is not None:
                    shepherd.maybe_submit_lazy(sheep, slurm, SubmitRules())
                    shepherd.commit()
                    record["job"] = sheep.current_job_id
            records.append(record)
    except (OSError, ValueError, RuntimeError, TypeError, sp.SubprocessError) as exc:
        note(f"error: {exc}")
        if output_mode(args) == JSON:
            emit(
                [],
                as_json={
                    "action": action,
                    "dry_run": dry_run,
                    "experiments": records,
                    "error": str(exc),
                },
            )
        return 1

    mode = output_mode(args)
    limit = args.limit or cap(MAX_ROWS, mode)
    shown = records[:limit]
    if mode == JSON:
        emit(
            [],
            as_json={
                "action": action,
                "dry_run": dry_run,
                "count": len(records),
                "shown": len(shown),
                "experiments": shown,
            },
        )
    else:
        prefix = "Would " if dry_run else ""
        lines = [f"{prefix}{action}: {len(records)} experiment(s)"]
        for record in shown:
            suffix = f" -> job {record['job']}" if record.get("job") else ""
            if "partition" in record:
                suffix += f"  partition={record['partition']}"
            lines.append(f"{record['sig']}  {action}{suffix}")
        if len(shown) < len(records):
            lines.append(f"... {len(records) - len(shown)} more (all selected XPs were processed)")
        emit(lines, mode=mode)
    return 0
