"""Profile OmegaConf costs on the Audium grid without changing either library."""
import argparse
from collections import Counter
from copy import deepcopy
import cProfile
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import pstats
import random
import statistics
import tempfile
from time import perf_counter
from unittest.mock import patch
import warnings

from omegaconf import OmegaConf

from .benchmark import assert_equal, collect_grid, make_main, settings_from_snapshot, snapshot


def shape(value):
    counts = Counter()
    if isinstance(value, dict):
        counts["dict_nodes"] += 1
        for child in value.values():
            counts.update(shape(child))
    elif isinstance(value, list):
        counts["list_nodes"] += 1
        for child in value:
            counts.update(shape(child))
    else:
        counts["scalar_nodes"] += 1
        if isinstance(value, str) and "$" + "{" in value:
            counts["interpolated_values"] += 1
    return counts


def time_stages(stages, repeat):
    samples = {name: [] for name in stages}
    rng = random.Random(0)
    for function, inputs in stages.values():
        for value in inputs:
            function(value)
    for _ in range(repeat):
        names = list(stages)
        rng.shuffle(names)
        for name in names:
            function, inputs = stages[name]
            start = perf_counter()
            for value in inputs:
                function(value)
            samples[name].append(1000 * (perf_counter() - start) / len(inputs))
    return {
        name: {"median_ms_per_config": statistics.median(values), "samples_ms_per_config": values}
        for name, values in samples.items()
    }


def profile_stage(function, inputs, repeat, path):
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(repeat):
        for value in inputs:
            function(value)
    profiler.disable()
    profiler.dump_stats(str(path))
    stats = pstats.Stats(profiler)
    calls = len(inputs) * repeat
    functions = []
    for (filename, lineno, name), (primitive, total, own, cumulative, _) in stats.stats.items():
        functions.append({
            "file": filename, "line": lineno, "function": name,
            "calls_per_config": total / calls,
            "primitive_calls_per_config": primitive / calls,
            "self_ms_per_config": 1000 * own / calls,
            "cumulative_ms_per_config": 1000 * cumulative / calls,
            "self_percent": 100 * own / stats.total_tt,
        })
    return {
        "profiled_ms_per_config": 1000 * stats.total_tt / calls,
        "calls_per_config": stats.total_calls / calls,
        "functions": sorted(functions, key=lambda item: item["self_ms_per_config"], reverse=True),
    }


def pipeline_components(main, rows, repeat):
    totals = Counter()

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                totals[name] += perf_counter() - start
        return wrapper

    with patch.object(main, "_get_config", timed("experiment_config", main._get_config)), \
            patch.object(main, "_get_base_config", timed("group_base_config", main._get_base_config)), \
            patch.object(main, "_get_delta", timed("compare_configs", main._get_delta)):
        start = perf_counter()
        for _ in range(repeat):
            for args in rows:
                main.get_xp(args)
        elapsed = perf_counter() - start
    totals["other"] = elapsed - sum(totals.values())
    return {
        name: {"mean_ms_per_config": 1000 * value / (repeat * len(rows)), "percent": 100 * value / elapsed}
        for name, value in totals.items()
    }


def run(args):
    import omegaconf

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dora-omegaconf-profile-") as temporary:
        root = Path(temporary)
        digests = snapshot(args.audium, root, args.grid)
        main = make_main(root / "config", settings_from_snapshot(root, None),
                         "parser", cache_group_bases=False)
        source = (root / "audiocraft/grids" / (args.grid.replace(".", "/") + ".py")).read_text()
        rows = collect_grid(source, main, True)
        raw = [main.parser.compose(argv) for argv in rows]
        configs = [main.parser.compose_config(argv) for argv in rows]
        bases = [main._get_base_config(argv)[0] for argv in rows]
        pairs = list(zip(bases, configs))
        resolved = [OmegaConf.to_container(cfg, resolve=True) for cfg in configs]
        expected = [main.get_xp(argv) for argv in rows]

        def compare(pair):
            return main._get_delta(*pair)

        stages = {
            "parser_dict": (main.parser.compose, rows),
            "omegaconf_create": (OmegaConf.create, raw),
            "omegaconf_create_resolved_input": (OmegaConf.create, resolved),
            "omegaconf_create_no_deepcopy_flag": (
                lambda value: OmegaConf.create(value, flags={"no_deepcopy_set_nodes": True}), raw),
            "set_struct": (lambda cfg: OmegaConf.set_struct(cfg, True), configs),
            "deepcopy_dict": (deepcopy, raw),
            "deepcopy_dictconfig": (deepcopy, configs),
            "to_container_unresolved": (lambda cfg: OmegaConf.to_container(cfg, resolve=False), configs),
            "to_container_resolved": (lambda cfg: OmegaConf.to_container(cfg, resolve=True), configs),
            "delta_existing_configs": (compare, pairs),
            "get_xp": (main.get_xp, rows),
        }
        timings = time_stages(stages, args.repeat)
        for name, result in timings.items():
            print(f"{name}: {result['median_ms_per_config']:.4f} ms/config", flush=True)
        components = pipeline_components(main, rows, args.repeat)

        # A measured control, local to this benchmark object. Cache a base used
        # only for reading during comparison; validate returned signatures/deltas.
        original_base = main._get_base_config
        cache = {}

        def cached_base(argv):
            key = tuple(arg for arg in argv if arg.partition("=")[0] in main._config_groups)
            if key not in cache:
                cache[key] = original_base(argv)
            config, delta = cache[key]
            return config, list(delta)  # get_xp mutates the returned delta list

        with patch.object(main, "_get_base_config", cached_base):
            for argv, reference in zip(rows, expected):
                actual = main.get_xp(argv)
                assert actual.sig == reference.sig
                assert_equal(actual.delta, reference.delta)
                assert_equal(OmegaConf.to_container(actual.cfg, resolve=True),
                             OmegaConf.to_container(reference.cfg, resolve=True))
            caching = time_stages({"get_xp_reuse_group_base": (main.get_xp, rows)}, args.repeat)
        timings.update(caching)
        print(f"get_xp_reuse_group_base: {timings['get_xp_reuse_group_base']['median_ms_per_config']:.4f} ms/config",
              flush=True)

        profiles = {}
        for name in ("omegaconf_create", "delta_existing_configs", "get_xp"):
            fn, inputs = stages[name]
            profiles[name] = profile_stage(fn, inputs, args.profile_repeat, destination / f"{name}.prof")
            print(f"Profile saved: {name}", flush=True)

        shapes = [shape(value) for value in raw]
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "omegaconf": omegaconf.__version__,
            "grid": args.grid, "configs": len(rows), "repeat": args.repeat,
            "profile_repeat": args.profile_repeat,
            "average_config_shape": {
                key: statistics.mean(count[key] for count in shapes)
                for key in sorted(set().union(*(count.keys() for count in shapes)))
            },
            "unique_group_bases": len(cache),
            "timings": timings, "get_xp_components": components, "profiles": profiles,
            "input_sha256": digests,
            "method": "Temporary config snapshot; recorded explorer and synthetic checkpoints; no training import. "
                      "Wall timings are unprofiled medians of shuffled batches after warmup. "
                      "cProfile self times/call counts locate hotspots; cumulative times overlap and "
                      "profiling overhead means profiled milliseconds are not production timings. "
                      "The group-base reuse control patches only the benchmark instance and checks all 20 outputs. "
                      "Lookup-only stages use existing warmed DictConfigs; complete get_xp constructs fresh ones.",
        }
        (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audium", type=Path, default=Path.home() / "projs/audium3")
    parser.add_argument("--grid", default="arflow.phonon1_fast2_langs")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("dora_parser/omegaconf_profile"))
    args = parser.parse_args()
    if min(args.repeat, args.profile_repeat) < 1:
        parser.error("repeat counts must be positive")
    warnings.simplefilter("ignore")
    run(args)


if __name__ == "__main__":
    main()
