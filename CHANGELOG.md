# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0a1] - TBD

Faster, quieter, and readable without importing your project.

- `import dora` no longer pulls in torch, hydra, omegaconf, submitit, treetable
  or retrying: 1.93s down to 0.13s, and `dora --help` 2.14s down to 0.36s.
  `jsonable` no longer forces a torch import on every XP construction.
- Added `dora status`, `dora metrics`, `dora log`, `dora why` and `dora plan`:
  read-only commands with capped, uncoloured, `--json`-able output, for looking
  at experiments without wading through treetables and multi-megabyte logs.
  `dora plan` also warns which experiments a real launch would cancel.
- Added an optional `dora.toml` for static project settings, so read-only
  commands need not import the training package. Supports `${env:VAR}`
  interpolation and `[[dora.dir_probe]]` for per-cluster experiment directories.
  Also fixes `dora` only working from the repository root.
- Added `main.get_existing_xp_from_sig()`, which reads what an experiment stored
  rather than recomposing its config. Faster, and it still works once the config
  files have moved on -- on one real project it recovered 301 experiments that
  could no longer be loaded at all. `init_xp` now persists the delta to make
  this possible.
- `--dry_run` no longer writes to the Dora directory. `Shepherd` gained a
  `read_only` mode which skips the orphan check, so reading job state can no
  longer cancel a job.
- `hydra.main` no longer leaves `GlobalHydra` initialized, so several XPs can
  run in one process.
- Sped up config group listing by suppressing Hydra's internal deepcopies
  (451ms to 153ms, at import time on every invocation).
- Added `dora.tests.golden`, a signature regression harness.
- Added `templates/SKILL.md`, a Claude Code skill for driving Dora.
- Minimum supported Python is now 3.11, so `dora.toml` parsing uses the stdlib
  `tomllib` with no third-party fallback.
- `dora --help` now describes every command -- `grid`, `launch`, `info`,
  `import` and `export` were listed but undocumented. `launch`, `info`,
  `import` and `export` are marked deprecated and listed last; they still work.

## [0.1.13] - 2026-09-09

This is the first release of the [kyutai-labs](https://github.com/kyutai-labs/dora) fork
of Dora, which has diverged from [facebookresearch/dora](https://github.com/facebookresearch/dora).

Adding dependent jobs. E.g., use `launcher.slurm_(dependents=5)`. Incompatible with
job arrays.

Adding possiblity to force the initialization of distributed even when world size=1 by setting
the `DORA_FORCE_DISTRIB=1` env variable. Always export LOCAL_RANK when running with `dora run`.

Not longer store the XP in the _SubmitItTarget in order to avoid potential pickling errors.

Adding `post_git_save_commands` to run commands from the clone of the repo when using git save.

Adding support for srun args.

Adding `nodelist` slurm param to restrict a job to a given list of nodes.

Adding `python` slurm param to let submitit use an alternative python interpreter.

Experimental support for jobs without GPUs (`gpus=0`), in which case no `gres` is requested.

Fixing issue with job array crashing.

Fixed docker support through `force_chdir` param.
Improved docker support with `container_chdir`.

Added option to run code locally. Changed Hydra flags so that only rank 0 logs the `.hydra/*.yaml` files

Removed lightning.

Added json files to easily get the job id.

No longer calling `scontrol show hostnames` at boot time, using submitit's nodelist parser
instead. Torch is now imported lazily by `dora.distrib`, which speeds up grid files.

Fixed submission happening from the wrong folder when using git save, and a related bug
with job arrays. Better error message when a local code clone exists but its tar file is missing.

Packaging moved from `setup.py`/`MANIFEST.in` to `pyproject.toml`, using the hatchling
build backend. The `examples` package is no longer installed alongside `dora`.
The minimum supported Python is now explicitly 3.10.

## [0.1.12] - 2023-05-23

Fixed bug with PL (Thanks @kingjr).

Added support for the Azure cluster (thanks @JadeCopet).

Fixed local rank bug.

Minor speed improvement if processing a lot of files with `to_absolute_path`.

Added `qos`, and `account` slurm params.

## [0.1.11] - 2022-09-22

Use job id based seed to avoid systematic failures with port allocation for distributed.

Remove automatic export of WORLD_SIZE inside submitit job target,
use `dora.distrib.set_distrib_env` if you relied on it.

Fixed version_base parameter support that appeared in Hydra.

## [0.1.10] - 2022-06-09

Updated and simplified PyTorch Lightning distributed integration.
Improved overall integration with PL, in particular with PLLogProgress and simplified
Dora logger.

Adding HiPlot support out of the box.

Fixed bug with nested grid searches.

Set `use_rendezvous=False` by default.

More reliable passing of arguments of Hydra (before, setting None would actually fail). I hope this wont break any existing XP sig...

Allow for empty `mem` constraint in Slurm.

Fixing `callbacks` default value in PL.

Extra "keys" in Hydra config files are now allowed (i.e. overrides with `+something=12`).

The package where Dora looks for grids can be customized, in Hydra with `dora.grid_package` in the base config or passing `grid_package='...'` to `argparse_main`.

Better doc for launcher API.

Fix dict support with Hydra. Okay it is time that I release a new version now...

## [0.1.9] - 2022-02-28

Reliable rmtree used to avoid `--clear` being blocked by some locking issues on NFS.

Fix bug with PL.

Early deletion of rendezvous file to avoid errors on job requeue. This might actually lead to
bugs in the future as this is not officially supported but from a discussion with PyTorch engineers,
"it should be okay".

Actually, because rendezvous file are not that reliable, added using Slurm to find
the master addr. Port is decided based on the XP signature (running twice the same XP on the
same machine will crash, but anyway this is probably a bad idea). Set `dora.use_rendezvous: false` to test out. This will soon become the default value.


## [0.1.8] - 2021-12-30

Always export RANK and WORLD_SIZE as env variable, so that they can be consumed by Hydra config
resolver.

`dora.log.LogProgress.update` now returns True if logging will happen at the end of the iteration.

Add silent option for grid API, which suppress all printing.

Adding `process_sheep` method in Explorer, that can replace `process_history` and provide access to the sheep and XP (`sheep.xp`)
in order to allow for processing that can depend on the config of the XP.

Automatically simplifies argv list for Hydra experiments when the same parameter is repeated multiple times.

Better error message when making a typo in the grid name. Always show the traceback when getting an
import error.

Easier sharing of XP hyper params. Added `import`/`export` command to easily share XP hyper-params in text form.
Added shared repository option (`shared` option in Dora config). No metrics or
checkpoints can be shared, this is still a bit dangerous, but this will act as a shared
database for mappings from SIG -> hyper params, so that you can just pass a SIG
to your teammate and launch the same XP. See [the README section on sharing](https://github.com/facebookresearch/dora/blob/main/README.md#sharing-xps) for more details.

## [0.1.7] - 2021-11-08

Adding support for type arrays.

Disabling automatic loading of PyTorch Lightning if installed, as this trigger
a warning with distutils/setuptools.

## [0.1.6] - 2021-10-20

Add py.typed file to source distribution.
Fixed bug in LogProgress for very slow speed.


## [0.1.5] - 2021-09-29

Added possiblity to log always the time per iteration rathen than iterations per seconds.
[PR](https://github.com/facebookresearch/dora/pull/10).

Fixed a bug with `DoraConf`, making sure that the `dir` attribute is always
absolute, even when reset after creation, which could happen when using Hydra
with a relative path for `dora.dir`.


## [0.1.4] - 2021-09-15

Initial release (first versions were private betas).
