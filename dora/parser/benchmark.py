"""Differential benchmark using the trusted Audium grid's actual explorer body.

No training imports, scheduler, experiment-directory reads, or writes to Audium.
Checkpoint existence is simulated both ways to cover every branch of this grid.
"""

import argparse
import ast
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import tomllib
import warnings
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import SimpleNamespace

from . import ConfigParser


def assert_equal(a, b, path=""):
    """Compare order and scalar types too: both can affect Dora signatures."""
    if type(a) is not type(b):
        raise AssertionError(f"{path}: different types: {type(a)} / {type(b)}")
    if isinstance(a, dict):
        if list(a) != list(b):
            raise AssertionError(f"{path}: different keys/order")
        for key in a:
            assert_equal(a[key], b[key], path + "." + str(key))
    elif isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: different list lengths")
        for index, (left, right) in enumerate(zip(a, b)):
            assert_equal(left, right, f"{path}[{index}]")
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def snapshot(audium, destination, grid):
    files = list((audium / "config").rglob("*.yaml"))
    files += list((audium / "config").rglob("*.yml"))
    files += [audium / "dora.toml", audium / "audiocraft/grids" / (grid.replace(".", "/") + ".py")]
    digests = {}
    for source in files:
        relative = source.relative_to(audium)
        data = source.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digests[str(relative)] = sha256(data).hexdigest()
    # Fail if a concurrent edit changed the inputs while taking the snapshot.
    for name, digest in digests.items():
        if sha256((audium / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input changed during snapshot: {name}; rerun")
    return digests


def make_main(config_dir, settings, backend, *, cache_group_bases=True):
    from dora.conf import DoraConfig
    from dora.hydra import HydraMain

    class BenchmarkMain(HydraMain):
        def __init__(self):
            self.config_name = "config"
            self.full_config_path = config_dir
            self._job_name = "train"
            self.hydra_kwargs = {"version_base": "1.1"}
            self.parser = ConfigParser(config_dir)
            self._base_cfg = self._get_config()
            self._config_groups = [
                str(path.relative_to(config_dir)) for path in config_dir.rglob("*") if path.is_dir()
            ]
            self.dora = DoraConfig(dir=settings["dir"], exclude=list(settings["exclude"]))
            self.dora.exclude += ["dora.*", "slurm.*"]

        def _get_config(self, overrides=()):
            if backend == "hydra":
                return super()._get_config(list(overrides))
            return self.parser.compose_config(overrides)

        def _get_base_config(self, overrides=()):
            if backend == "hydra":
                return super()._get_base_config(list(overrides))
            # Match Dora's group-only base/delta behavior; leave all signature
            # computation in the existing HydraMain and XP implementations.
            kept, delta = [], []
            for arg in overrides:
                key, _, value = arg.partition("=")
                if key in self._config_groups:
                    kept.append(arg)
                    delta = [(k, v) for k, v in delta if k != key]
                    delta.append((key, value))
            if cache_group_bases:
                return self.parser.compose_base_config(kept), delta
            return (self.parser.compose_config(kept) if kept else self._base_cfg), delta

    return BenchmarkMain()


class _CheckpointPath:
    def __init__(self, path, present):
        self.path = PurePosixPath(path)
        self.present = present

    def __truediv__(self, name):
        return _CheckpointPath(self.path / name, self.present)

    def exists(self):
        return self.present

    def __str__(self):
        return str(self.path)


def collect_grid(source, main, checkpoints):
    from dora.hydra import _simplify_argv

    rows = []

    class Launcher:
        def __init__(self, argv=()):
            self._argv = list(argv)

        def bind_(self, *args, **kwargs):
            for arg in (*args, kwargs):
                self._argv.extend(main.value_to_argv(arg))
            return self

        def bind(self, *args, **kwargs):
            return Launcher(self._argv).bind_(*args, **kwargs)

        def slurm_(self, **kwargs):
            return self

        def __call__(self, *args, **kwargs):
            rows.append(_simplify_argv(self.bind(*args, **kwargs)._argv))

    def get_xp(argv):
        xp = main.get_xp(argv)
        return SimpleNamespace(
            argv=xp.argv, sig=xp.sig, folder=_CheckpointPath(xp.folder, checkpoints)
        )

    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "explorer"
    ]
    if len(functions) != 1:
        raise ValueError("Expected one explorer function")
    function = functions[0]
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"train": SimpleNamespace(main=SimpleNamespace(get_xp=get_xp))}
    # The benchmark executes trusted local source captured above.
    exec(compile(module, "<captured Audium explorer>", "exec"), namespace)  # noqa: S102
    namespace["explorer"](Launcher())
    return rows


def settings_from_snapshot(root, xp_root):
    from omegaconf import OmegaConf

    settings = tomllib.loads((root / "dora.toml").read_text())["dora"]
    base = ConfigParser(root / "config").compose_config()
    if "dora" in base:
        settings.update(OmegaConf.to_container(base.dora, resolve=True))
    if xp_root is None:

        def expand(text):
            return re.sub(r"\$\{env:([^}]+)\}", lambda m: os.environ.get(m[1], ""), text)

        xp_root = expand(settings.get("dir", ""))
        if not xp_root:
            for probe in settings.get("dir_probe", []):
                if Path(probe["probe"]).exists():
                    xp_root = expand(probe["dir"])
                    break
    if not xp_root:
        raise ValueError("Cannot infer the experiment root; pass --xp-root")
    return {"dir": xp_root, "exclude": settings.get("exclude", [])}


def measure(fn, cases, repeat):
    samples = []
    for _ in range(repeat):
        start = perf_counter()
        for args in cases:
            fn(args)
        samples.append((perf_counter() - start) * 1000 / len(cases))
    return {
        "median_ms_per_config": statistics.median(samples),
        "min_ms_per_config": min(samples),
        "samples_ms_per_config": samples,
    }


def cold_process(config_dir, args, backend, repeat):
    code = """
import sys, json, time, warnings
warnings.simplefilter('ignore')
started = time.perf_counter()
root, args, backend = json.loads(sys.argv[1])
if backend == 'hydra':
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(root, version_base='1.1'):
        cfg = compose('config', args)
else:
    from dora.parser import ConfigParser
    parser = ConfigParser(root)
    cfg = parser.compose_config(args) if backend == 'omegaconf' else parser.compose(args)
print((time.perf_counter() - started) * 1000)
"""
    samples = []
    for _ in range(repeat):
        output = subprocess.check_output(
            [sys.executable, "-B", "-c", code, json.dumps([str(config_dir), args, backend])],
            text=True,
        )
        samples.append(float(output))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def audit_configs(config_dir):
    """Check all YAML documents and every single-solver choice in the snapshot."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from .yaml_loader import load_yaml

    count = 0
    for path in sorted(config_dir.rglob("*.yaml")):
        expected = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        actual = load_yaml(path.read_text())
        assert_equal({} if actual is None else actual, expected, str(path.relative_to(config_dir)))
        count += 1
    parser = ConfigParser(config_dir)
    matched, hydra_rejected, mismatches = [], {}, {}
    with initialize_config_dir(str(config_dir), version_base="1.1"):
        for path in sorted((config_dir / "solver").rglob("*.yaml")):
            arg = "solver=" + str(path.relative_to(config_dir / "solver").with_suffix(""))
            try:
                expected = OmegaConf.to_container(compose("config", [arg]), resolve=False)
            except Exception as exc:  # noqa: BLE001 -- compare all backend failures
                hydra_rejected[arg] = type(exc).__name__ + ": " + str(exc).splitlines()[0]
                continue
            try:
                assert_equal(parser.compose([arg]), expected, arg)
            except Exception as exc:  # noqa: BLE001 -- compare all backend failures
                mismatches[arg] = str(exc)
            else:
                matched.append(arg)
    if mismatches:
        raise AssertionError(json.dumps(mismatches, indent=2))
    print(
        f"Audit: {count} YAML documents and {len(matched)} solver configs match; "
        f"Hydra rejects {len(hydra_rejected)} other solver choices.",
        flush=True,
    )
    return {"yaml_matched": count, "solvers_matched": matched, "hydra_rejected": hydra_rejected}


def run(audium, grid, repeat, cold_repeat, xp_root, audit_solvers=False):
    import hydra
    import omegaconf
    import yaml  # type: ignore[import-untyped]
    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory(prefix="dora-parser-") as temporary:
        root = Path(temporary)
        digests = snapshot(audium, root, grid)
        config_dir = root / "config"
        audit = audit_configs(config_dir) if audit_solvers else None
        settings = settings_from_snapshot(root, xp_root)
        oracle = make_main(config_dir, settings, "hydra")
        fast = make_main(config_dir, settings, "parser")
        uncached = make_main(config_dir, settings, "parser", cache_group_bases=False)
        source = (root / "audiocraft/grids" / (grid.replace(".", "/") + ".py")).read_text()
        rows = []
        branch_counts = {}
        for present in (False, True):
            expected = collect_grid(source, oracle, present)
            actual = collect_grid(source, fast, present)
            assert_equal(actual, expected)
            branch_counts["checkpoints_present" if present else "checkpoints_absent"] = len(actual)
            for row in actual:
                if row not in rows:
                    rows.append(row)
        signatures = []
        for index, args in enumerate(rows):
            expected = oracle.get_xp(args)
            actual = fast.get_xp(args)
            previous = uncached.get_xp(args)
            assert_equal(actual.delta, previous.delta)
            assert actual.sig == previous.sig
            for resolve in (False, True):
                assert_equal(
                    OmegaConf.to_container(actual.cfg, resolve=resolve),
                    OmegaConf.to_container(expected.cfg, resolve=resolve),
                )
            assert_equal(actual.delta, expected.delta)
            assert actual.sig == expected.sig
            signatures.append(actual.sig)
        print(
            f"Parity: {len(rows)} configs, resolved configs, deltas and signatures match. "
            f"{branch_counts}",
            flush=True,
        )
        timings = {}
        stages = {
            "hydra_compose": oracle._get_config,
            "parser_dict_warm": fast.parser.compose,
            "parser_omegaconf_warm": fast.parser.compose_config,
            "parser_dict_fresh_instance": lambda args: ConfigParser(config_dir).compose(args),
            "hydra_get_xp": oracle.get_xp,
            "parser_get_xp": fast.get_xp,
            "parser_get_xp_uncached_base": uncached.get_xp,
        }
        for name, fn in stages.items():
            timings[name] = measure(fn, rows, repeat)
            print(f"{name}: {timings[name]['median_ms_per_config']:.3f} ms/config", flush=True)
        cold = {}
        for backend in ("hydra", "dict", "omegaconf"):
            cold[backend] = cold_process(config_dir, rows[0], backend, cold_repeat)
            print(
                f"fresh process import + compose ({backend}): {cold[backend]['median_ms']:.3f} ms",
                flush=True,
            )
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "source": str(audium),
            "grid": grid,
            "python": platform.python_version(),
            "hydra": hydra.__version__,
            "omegaconf": omegaconf.__version__,
            "pyyaml": yaml.__version__,
            "libyaml": yaml.__with_libyaml__,
            "repeat": repeat,
            "branch_counts": branch_counts,
            "audit": audit,
            "unique_configs": len(rows),
            "parity": {"unresolved": True, "resolved": True, "deltas": True, "signatures": True},
            "signatures": signatures,
            "settings": settings,
            "timings": timings,
            "fresh_process_import_and_compose": cold,
            "input_sha256": digests,
            "argv": rows,
            "method": "Temporary snapshot; simulated checkpoint absence/presence; "
            "no training imports. "
            "Warm timings include default cache dependency stat checks. "
            "Fresh instance includes YAML loading; fresh process includes imports "
            "but excludes interpreter startup. "
            "get_xp uses Dora's existing OmegaConf comparison and XP signature code. "
            "parser_get_xp caches read-only group bases with dependency checks; "
            "parser_get_xp_uncached_base rebuilds their DictConfigs.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audium", type=Path, default=Path.home() / "projs/audium3")
    parser.add_argument("--grid", default="arflow.phonon1_fast2_langs")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cold-repeat", type=int, default=3)
    parser.add_argument(
        "--xp-root", help="Experiment path used for hypothetical continuation overrides"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--audit-solvers", action="store_true", help="Also compare all YAML and solver choices"
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.cold_repeat < 1:
        parser.error("repeat counts must be positive")
    warnings.simplefilter("ignore")
    result = run(
        args.audium.expanduser().resolve(),
        args.grid,
        args.repeat,
        args.cold_repeat,
        args.xp_root,
        args.audit_solvers,
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
