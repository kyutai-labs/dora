---
name: dora-experiments
description: Inspect, launch and debug Dora experiments — signatures, grids, job state, metrics and failures. Use when asked about an experiment, a signature (an 8-hex-character id), a grid, why a run died, or what a training job is doing.
---

# Working with Dora experiments

Dora identifies every experiment by a **signature**: 8 hex characters derived
from the config, not from a name you choose. Everything below is addressed by
that signature, by a grid name, or by a Slurm job id.

## Read state with the inspection commands, not by reading files

```bash
dora status <grid|sig|jobid>...     # one line per experiment
dora metrics <sig> [--every 20]     # downsampled history
dora log <sig> [--tail 40] [--grep] # log, decoloured
dora why <sig>                      # what killed it
```

All four are read-only. Because their output is not going to a terminal, they
default to a capped, uncoloured rendering; `--json` gives one object instead,
and `--pretty` the colourised form a person would want. They are the right tool
because the alternatives are enormous: a real `history.json` is hundreds of KB,
a `solver.log.*` is megabytes and full of ANSI escapes, and `dora grid` prints a
wrapped treetable that costs tens of KB to say what `dora status` says in two.

**Do not `cat` anything under the experiment directory.** If you find yourself
wanting to, the answer is a flag on one of the commands above.

Start with `dora why` when something failed; fall back to `dora log --grep` only
if it reports no known signature.

## What a signature is, and when it changes

The signature is `sha1(sorted(delta))[:8]`, where the delta is the difference
between an experiment's config and the project's defaults. Consequences worth
internalising:

- **Argument order does not matter.** The delta is sorted before hashing.
- **Setting a parameter to its default changes nothing.** It does not enter the
  delta, so it does not enter the signature.
- **Parameters listed in `dora.exclude` never affect it.** That is how you change
  logging, workers or debug flags without orphaning a run. `dora.*` and `slurm.*`
  are always excluded, so the number of GPUs and the partition never matter.
- **Changing the config files can move signatures of past experiments.** The
  delta is computed against the defaults *as they are now*.

That last point is why `dora status` and friends read what an experiment stored
rather than recomputing it.

## Grids

A grid is a Python file, `<package>/grids/<a>/<b>.py`, exposing `explorer`:

```python
@SomeExplorer
def explorer(launcher):
    launcher.slurm_(gpus=8, partition="...")   # trailing _ mutates in place
    launcher.bind_({"solver": "...", "optim.lr": 1e-4})
    sub = launcher.bind({"model.dim": 512})    # no _ returns a new launcher
    sub()                                      # schedules one experiment
```

`dora grid a.b` maps to `<package>/grids/a/b.py`. **Grid files are branch-local
while the experiment directory accumulates grids from every branch ever
launched**, so a grid you can see in `dora status` may not exist on your branch.

```bash
dora grid a.b --dry_run --compact   # resolve to signatures, change nothing
dora grid a.b                       # actually launch
```

`--dry_run` writes nothing at all. `--dry_run --init` additionally registers the
signatures so they can be referenced later, and nothing else.

**Launching a grid cancels experiments it no longer produces.** Anything
currently symlinked under the grid that the edited explorer stops emitting is
cancelled if still running, and dropped from the grid if already finished.
`--dry_run --compact` names both sets before you launch.

## Reading an experiment from Python

```python
from mypackage.train import main

xp = main.get_existing_xp_from_sig(sig)   # what it stored: fast, and works
                                          # even if the config files moved on
xp = main.get_xp_from_sig(sig)            # recomposed from today's configs: slow,
                                          # and fails for older experiments
xp.folder, xp.argv, xp.delta, xp.cfg
```

Prefer `get_existing_xp_from_sig`. `main.get_xp_from_argv` does not exist -- the
function is `main.get_xp(argv)`.

## Do not do these

- Do not build a `Shepherd` just to read state. Its constructor runs an orphan
  check that can `scancel` live jobs. Pass `read_only=True` if you must.
- Do not `--clear` or cancel a grid without being asked to; both are destructive
  and `--clear` deletes checkpoints.
