# dora_parser (experimental)

The implementation now lives in `dora.parser` and ships with Dora.
Enable it with `[project] use_fast_parser = true` in `dora.toml` (false by
default). This folder retains benchmark/profile scripts, historical reports,
and compatibility imports. See the root README for integration details.

The dictionary API needs Python 3.11+ and PyYAML. It does not import Hydra,
OmegaConf, or ANTLR. PyYAML uses libyaml when available and otherwise falls back
to its Python loader. The optional DictConfig API needs OmegaConf. Hydra is
only needed for differential tests and benchmarks.

## Try it

Run from the Dora checkout using its existing environment, without syncing
dependencies:

~~~python
from dora_parser import ConfigParser

parser = ConfigParser("/data/home/alex/projs/audium3/config")
overrides = [
    "solver=arflow/tts2",
    "conditioner=tts_pocket",
    "dataset.batch_size=48",
    "+dataset.train.augmenter=basic",
]

raw = parser.compose(overrides)          # independent dict; interpolations stay strings
print(raw["dataset"]["batch_size"])

cfg = parser.compose_config(overrides)   # OmegaConf DictConfig with struct mode
print(cfg.dataset.batch_size)
print(cfg.conditioners.speaker_wavs.encodec.compression_model_checkpoint)
~~~

There is also a one-shot helper:
`compose(config_dir, overrides=(), config_name="config")`.
Reuse a ConfigParser instance to benefit from caching across a grid.

~~~bash
uv run --no-sync python -B -m dora_parser \
  --config-dir /data/home/alex/projs/audium3/config \
  solver=arflow/tts2 dataset.batch_size=48
~~~

Add `--resolve` to resolve interpolation through OmegaConf;
`--format json` changes the output format.

## Supported subset

- YAML mappings, scalars, lists, scientific notation, dates retained as strings,
  mandatory missing values (`???`), and duplicate-key checks.
- Defaults lists, explicit or implicit `_self_`, nested groups, absolute
  and relative includes, and parent-relative includes within the config root.
  For a selection like `solver=arflow/tts2`, relative defaults start at `solver/`.
- Defaults `override /group: choice`, `optional group: choice`, and null
  placeholders. Later subtrees can override earlier group selections, and CLI
  selections take precedence over defaults overrides.
- Package headers and group relocation (`group@package=choice`), including
  `_global_`, Audium's historical `__global__` spelling, and `_here_`.
- Dotted overrides, `+` additions, `++` replacement/addition,
  `~key[=value]` deletion, and dotted list indices. Dictionary overrides
  merge recursively; lists replace. Unknown keys require `+`.
- Hydra-style CLI scalar typing, quotes, escapes, lists, and dictionaries with
  unquoted keys. CLI parsing uses a recursive descent parser, not YAML or JSON:
  `yes` and `01` are strings; `1e-5` is a float. Quoted Unicode escape
  sequences remain literal, matching the current Dora/Hydra serializer.
- Lazy interpolation through the optional OmegaConf boundary. Registered
  OmegaConf resolvers remain available there; the dictionary API preserves
  expressions rather than resolving them.

Unsupported features raise ConfigError or UnsupportedFeature: multirun/sweep
functions, multiple options in a defaults group, interpolated defaults choices,
legacy `_group_`/`_name_` package substitutions, Hydra runtime settings,
search-path plugins, and updates/merges that traverse interpolated containers.
ConfigStore/structured schemas and Hydra instantiation are outside the parser API.
The opt-in Dora integration handles logging, run directories, and saved config metadata.

This is a tested subset, not a complete implementation of Hydra's grammar or
all OmegaConf merge semantics. CLI `group=null` is rejected like Hydra;
use `~group` to remove a selection.

## Caching and isolation

Composition builds a defaults plan, merges plain containers, then applies value
overrides. Parsed YAML is cached per file, and up to 32 composed group bases are
cached per parser. The compose() and compose_config() methods return independent
configs so edits cannot leak into later experiments. The compose_base_config()
method instead returns a shared, read-only base.

Cached file dependencies (including missing optional files) are checked using
mtime, size, and inode on every call. Config edits invalidate affected bases.
`parser.clear_cache()` explicitly resets caches;
`ConfigParser(..., cache_size=0)` disables composed-base caching.
`check_files=False` skips dependency checks on cached bases and is only
appropriate for immutable inputs. Use one parser instance per thread.

CLI overrides apply in order. The core does not deduplicate repeated keys;
Dora's existing `_simplify_argv` handles that in the benchmark adapter.

### Reusing group-base configs

`parser.compose_base_config(overrides)` returns a shared, read-only OmegaConf
config containing only the selected groups. Ordinary value overrides are ignored,
so experiments that differ in learning rate, batch size, or data paths reuse
their group base:

~~~python
base = parser.compose_base_config(["solver=arflow/tts2", "optim.lr=1e-4"])
same = parser.compose_base_config(["solver=arflow/tts2", "optim.lr=2e-4"])
assert same is base

experiment = parser.compose_config(["solver=arflow/tts2", "optim.lr=2e-4"])
# experiment is independent and mutable; base rejects ordinary mutations.
~~~

The cache key includes the primary config name and ordered group-selection
arguments. Raw dictionaries and their lazily constructed DictConfigs share one
LRU cache, bounded by `cache_size` (32 by default). File edits, deletions, and
newly available optional configs invalidate both representations.
`clear_cache()` clears both; `cache_size=0` disables their reuse.

Interpolation expressions remain unresolved. OmegaConf resolver caches are reset
on each base retrieval, preserving freshness even for resolvers registered with
`use_cache=True`. The returned base is borrowed: do not disable its readonly
flag or use it for training. Use `compose_config()` for a mutable config.

The opt-in `dora.hydra.HydraMain` integration uses this cache and returns a fresh
group-delta list for each experiment. Hydra remains the default backend.
`make_main(..., cache_group_bases=False)` in the benchmark reproduces the previous
behavior for comparison.

Measured on all 20 grid variants (median of three passes):

| get_xp path | ms/experiment |
| --- | ---: |
| Hydra | 194.16 |
| Prototype, rebuilding group bases | 30.21 |
| Prototype, cached group bases | 20.63 |

Caching saves **31.7%** of the prototype's get_xp time
(1.46x throughput). All 20 unresolved/resolved configs, deltas, and
signatures match Hydra. The cache regression tests cover file changes,
optional files, eviction, mutation protection, fresh resolver values, and
Dora signatures. The original prototype suite passed 118 tests; parser and
integration tests now live under `dora/tests`.

Raw results are generated locally in `group_base_cache_results.json` and ignored
by Git. Reproduce:

~~~bash
uv run --no-sync python -B -m dora_parser.benchmark \
  --repeat 3 --cold-repeat 3 --output dora_parser/group_base_cache_results.json
~~~

## Verification and speed

~~~bash
uv run --no-sync python -B -m pytest dora/tests/parser -q -p no:cacheprovider
uv run --no-sync python -B -m flake8 dora_parser --max-line-length 120
uv run --no-sync python -B -m dora_parser.benchmark \
  --audium /data/home/alex/projs/audium3 \
  --audit-solvers --repeat 3 --cold-repeat 3 \
  --output dora_parser/benchmark_results.json
~~~

The benchmark snapshots YAML, the target grid, and dora.toml into a temporary
directory and records input hashes. It compiles only the trusted grid's
undecorated explorer function, supplies a recording launcher and a minimal
train object, and simulates checkpoint presence/absence. It does not import
Audium training code, inspect live checkpoints, submit jobs, or write to Audium.

The `arflow.phonon1_fast2_langs` grid emits 10 pretraining variants when
checkpoints are absent and 20 variants when all continuation branches are
enabled. Every unique variant must match Hydra in dictionary order, scalar
types, unresolved and resolved values, Dora deltas, and signatures before timing
starts. The harness uses Dora's existing value serialization, config comparison,
exclusion filtering, and XP signature code. These are recompositions of the
snapshot, not comparisons against historical experiment files.

`--audit-solvers` also compares YAML typing throughout the config tree and
every solver selection that Hydra itself accepts.

Initial measurements, before group-base caching, on 2026-09-09 with Python 3.12.11, Hydra 1.3.6,
OmegaConf 2.3.1, and PyYAML 6.0.3 with libyaml. Median of three
passes over all 20 unique configs; warm timings include cache dependency checks.

| Operation | Time per config | Speedup against Hydra |
| --- | ---: | ---: |
| Hydra compose (DictConfig) | 100.75 ms | 1x |
| Parser compose (dict, warm) | 1.14 ms | 88.7x |
| Parser compose_config (DictConfig, warm) | 11.53 ms | 8.7x |
| Parser compose (dict, fresh instance) | 4.64 ms | 21.7x |
| Existing Hydra get_xp | 196.48 ms | 1x |
| Parser adapter get_xp | 30.96 ms | 6.3x |

Fresh-process import plus composition, excluding interpreter startup:
Hydra 303.6 ms,
parser dict 50.1 ms,
parser DictConfig 110.4 ms.

The wider audit matched 214 YAML documents and 73 solver configurations.
Hydra itself rejected 6 other solver choices, recorded in the report.
The generated `benchmark_results.json` contains samples, input hashes,
all 20 signatures, and captured arguments. Timings measure config work, not training startup.

The OmegaConf boundary is a substantial part of the remaining cost. Keeping
OmegaConf initially provides a straightforward route to faster composition
while preserving lazy interpolation and existing callers. Faster experiment
signature calculation would be a separate change: this benchmark keeps Dora's
existing comparison and hashing code.

The opt-in production integration also handles logging, run directories, and config
snapshots. These timings isolate composition and do not measure that runtime setup.

See [OMEGACONF_PROFILE.md](OMEGACONF_PROFILE.md) for a measured breakdown of
OmegaConf construction, access, and comparison costs.

## References

The behavior was checked against installed Hydra source, especially
[defaults-list construction](https://github.com/hydra-ecosystem/hydra/blob/main/hydra/_internal/defaults_list.py),
[config loading and overrides](https://github.com/hydra-ecosystem/hydra/blob/main/hydra/_internal/config_loader_impl.py),
the [override parser grammar](https://github.com/hydra-ecosystem/hydra/blob/main/hydra/grammar/OverrideParser.g4),
[lexer grammar](https://github.com/hydra-ecosystem/hydra/blob/main/hydra/grammar/OverrideLexer.g4),
and [package resolution](https://github.com/hydra-ecosystem/hydra/blob/main/hydra/core/default_element.py).
The prototype is an independent implementation; it does not vendor Hydra.

## Integrated benchmark

Run `uv run --no-sync python -B -m dora_parser.benchmark_integration` to exercise
the actual HydraMain constructor and opt-in path. It generates the Git-ignored
`integration_results.json`. The verification run recorded in this report
matched all 20 unique grid experiments (including
resolved configs, deltas, and signatures), all 214 YAML documents, and all 73
solver choices accepted by Hydra. Median warm get_xp: **20.5 ms with the fast
parser, 204.2 ms with Hydra**. Inputs are copied to /tmp; Audium is never modified.
