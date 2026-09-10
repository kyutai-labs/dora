"""Check the packaged HydraMain opt-in against Hydra on an Audium grid snapshot."""
import json
from pathlib import Path
import tempfile
import tomllib
from unittest.mock import patch

from omegaconf import OmegaConf

from dora.hydra import HydraMain
from dora.project import ProjectConfig
from .benchmark import (assert_equal, audit_configs, collect_grid, measure,
                        settings_from_snapshot, snapshot)


def task(cfg):
    raise AssertionError("Benchmark must never run training")


def run():
    grid = "arflow.phonon1_fast2_langs"
    with tempfile.TemporaryDirectory(prefix="dora-integration-") as temporary:
        root = Path(temporary)
        digests = snapshot(Path.home() / "projs/audium3", root, grid)
        settings = settings_from_snapshot(root, None)
        raw = tomllib.loads((root / "dora.toml").read_text())
        raw["dora"]["dir"] = settings["dir"]
        mains = []
        for enabled in (False, True):
            raw["project"]["use_fast_parser"] = enabled
            conf = ProjectConfig(root / "dora.toml", raw)
            task.__module__ = __name__
            with patch("dora.project.load", return_value=conf):
                mains.append(HydraMain(task, "config", str(root / "config"), version_base="1.1"))
        oracle, fast = mains
        source = (root / "audiocraft/grids/arflow/phonon1_fast2_langs.py").read_text()
        rows, counts = [], {}
        for present in (False, True):
            expected = collect_grid(source, oracle, present)
            actual = collect_grid(source, fast, present)
            assert_equal(actual, expected)
            counts[str(present)] = len(actual)
            for row in actual:
                if row not in rows:
                    rows.append(row)
        signatures = []
        for argv in rows:
            a, b = oracle.get_xp(argv), fast.get_xp(argv)
            assert a.sig == b.sig
            assert_equal(a.delta, b.delta)
            for resolve in (False, True):
                assert_equal(OmegaConf.to_container(a.cfg, resolve=resolve),
                             OmegaConf.to_container(b.cfg, resolve=resolve))
            signatures.append(a.sig)
        print(f"Integrated parity: {len(rows)} unique XPs, branches {counts}", flush=True)
        timings = {}
        for name, main in (("hydra", oracle), ("fast_parser", fast)):
            timings[name] = measure(main.get_xp, rows, 3)
            print(name, timings[name]["median_ms_per_config"], "ms/XP", flush=True)
        result = {"grid": grid, "branch_counts": counts, "unique_configs": len(rows),
                  "signatures": signatures, "timings": timings,
                  "audit": audit_configs(root / "config"), "input_sha256": digests,
                  "method": "Actual HydraMain constructor and get_xp; mocked ProjectConfig "
                            "enables the same TOML flag. Read-only Audium snapshot; simulated "
                            "checkpoint branches; no training or scheduling."}
    destination = Path(__file__).with_name("integration_results.json")
    destination.write_text(json.dumps(result, indent=2) + "\\n")
    return result


if __name__ == "__main__":
    run()
