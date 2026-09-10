# OmegaConf profiling on the Audium grid

Measured 2026-09-09, Python 3.12.11, OmegaConf 2.3.1, on all 20 variants of
`arflow.phonon1_fast2_langs`. No Dora, parser, OmegaConf, or Audium runtime code was changed.

The group-base cache suggested below now ships in the opt-in parser; see
[README.md](README.md#reusing-group-base-configs) for current results. This report
and its profiling command retain the uncached baseline for comparison.

## Finding

The main cost is constructing and accessing the full OmegaConf node tree.
Each scalar becomes a node with metadata, parent links, flags, and validation.
Construction routes each item through the general mutation machinery even
though this workload consists of ordinary untyped dictionaries and lists.

The average config has 73 dict nodes, 16 list nodes,
385.5 scalar nodes, and only 1.5 interpolation expressions.
One `OmegaConf.create(raw)` invokes approximately 123,750 function calls.

## Unprofiled timings

Medians of seven batches, each traversing all 20 variants. Batch order is
shuffled between repeats. Inputs and relevant caches are warmed first.
The create measurements begin with a prepared dict: YAML parsing, Hydra,
and the new parser are excluded from those measurements.

| Operation | ms/config |
| --- | ---: |
| Plain dictionary composition | 1.227 |
| OmegaConf.create from an already composed dict | 10.568 |
| Same construction with interpolation already resolved | 10.346 |
| Same construction with no_deepcopy_set_nodes=True | 10.686 |
| Set struct mode on existing configs | 0.130 |
| Deepcopy the ordinary dictionary | 0.193 |
| Deepcopy the DictConfig | 6.982 |
| Convert existing DictConfig to dict, resolve=False | 1.399 |
| Convert existing DictConfig to dict, resolve=True | 2.002 |
| Dora delta on existing DictConfigs | 7.138 |
| Complete parser-backed get_xp | 30.754 |
| get_xp control with reusable group base | 20.349 |

The interpolation-free input changes construction time by about
2.1%. Disabling node deepcopy has no meaningful benefit:
the creation profile makes no deepcopy calls when its inputs are ordinary
Python containers. The flag applies to assigning existing OmegaConf nodes.

## Construction hotspots

Call counts below are per config, from cProfile:

| Function | Calls/config | Role |
| --- | ---: | --- |
| `basecontainer.py:524 _set_item_impl` | 473.5 | General assignment path: validate, check flags/types, wrap |
| `omegaconf.py:994 _node_wrap` | 473.5 | Create DictConfig, ListConfig, or scalar node |
| `_utils.py:447 is_structured_config` | 1,568.5 | Check for dataclass/attrs schemas |
| `base.py:189 _get_flag` | 3,324.5 | Check inherited readonly/struct/convert/allow_objects flags |
| `base.py:152 _set_flag` | 949.0 | Temporarily set/restore flags during construction |

Exclusive profiler time attributes about 17% to isinstance/ABC checks, 12% to
flag handling, and 5% to import bookkeeping. Helpers repeatedly execute local
`from omegaconf import ...` statements, which still incur cached import
bookkeeping. These percentages are disjoint self-time totals, not nested
cumulative totals. Node allocation, validation, assignment, and other helper
calls account for the remainder.

The actual interpolation grammar parser is never called during construction
on this grid: its simple expressions pass the regex validation shortcut in
`omegaconf/_utils.py:_is_interpolation_string`.

## Comparison hotspots

A delta comparison invokes about 99,769 function calls:
800.5 item lookups and
393 membership tests per config. `key in cfg` calls
`_resolve_with_default`, so it validates and resolves the value. Dora then
looks up that same key again. Mandatory-missing checks, interpolation checks,
key normalization, and ABC instance checks recur for each lookup.

Membership tests account for 28.3% of the comparison's profiled cumulative time.
Actual grammar parsing occurs 5 times per comparison and accounts for only
5.4% of the profiled comparison time. Interpolation is a small share
for this grid, though it can dominate configs with many complex expressions.

## Complete get_xp breakdown

A separate unprofiled run instruments the three existing method boundaries.
These are mean stage times and add up to the whole run; lookup-only benchmarks
above use existing configs, whereas this path constructs fresh ones.

| Stage | ms/config | Share |
| --- | ---: | ---: |
| Build experiment DictConfig | 12.113 | 40.5% |
| Build group-base DictConfig | 9.560 | 31.9% |
| Compare configs | 7.366 | 24.6% |
| Argument processing, exclusions, XP construction, hashing, other | 0.886 | 3.0% |

## Next changes suggested by the measurements

1. Reuse the composed group base: all 20 variants have 1 unique base. The
   profiling-only control caches that private config, returns a fresh delta
   list, and reduces get_xp from 30.75 to 20.35 ms. All 20 resolved
   configs, deltas, and signatures were checked for equality. A production
   implementation would need the same file invalidation as the parser and
   protection against callers mutating the cached base.
2. Reduce repeated access during comparison. Membership plus item lookup
   repeats resolution. Comparing ordinary containers or using a carefully
   defined node-access path could avoid much of this work; preserving missing
   value, interpolation, and signature behavior needs explicit tests.
3. If optimizing OmegaConf itself, add a bulk construction path for plain
   untyped dict/list input. The general assignment and per-node flag machinery
   are the largest opportunity indicated here. This is a proposed direction,
   not an implemented or measured replacement.

## Reproduce

```bash
uv run --no-sync python -B -m dora_parser.profile_omegaconf
```

The runner snapshots configs and the grid into a temporary directory and
simulates checkpoint existence. It imports no training code and touches no
live experiment files. The only method patch is on its own benchmark object
for the base-reuse control; it is restored afterward.

The command generates `omegaconf_profile/results.json` and binary cProfile files
`omegaconf_create.prof`, `delta_existing_configs.prof`, and `get_xp.prof` in the
same directory. These generated artifacts are ignored by Git.

cProfile increases absolute runtime by roughly 3–4x here. Use the unprofiled
table for latency, and the profiles for call counts and hotspots. Cumulative
times include callees and must not be summed across nested functions.
