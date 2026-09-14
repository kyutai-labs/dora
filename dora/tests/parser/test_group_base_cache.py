import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ReadonlyConfigError

from dora.parser import ConfigError, ConfigParser


def write(root, name, content):
    path = root / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def root(tmp_path):
    write(
        tmp_path,
        "config",
        """defaults:
  - _self_
  - solver: small
  - optional extra: absent
lr: 0.01
optim: {momentum: 0.9}
values: [1, 2]
alias: ${lr}
environment: ${oc.env:DORA_PARSER_CACHE_TEST,unset}
""",
    )
    write(tmp_path, "common", "# @package _global_\nshared: 1\n")
    for name, width in (("small", 8), ("big", 16)):
        write(
            tmp_path,
            "solver/" + name,
            f"# @package _global_\ndefaults: [/common]\nwidth: {width}\n",
        )
    return tmp_path


def test_same_groups_reuse_base_but_experiments_stay_independent(root):
    parser = ConfigParser(root)
    base = parser.compose_base_config(["solver=big", "lr=0.1"])
    assert parser.compose_base_config(["lr=0.2", "solver=big"]) is base
    assert base.lr == 0.01
    assert base.width == 16
    assert OmegaConf.is_readonly(base)
    assert OmegaConf.is_struct(base)
    with pytest.raises(ReadonlyConfigError):
        base.lr = 7
    with pytest.raises(ReadonlyConfigError):
        base.optim.momentum = 0
    with pytest.raises(ReadonlyConfigError):
        base["values"].append(3)
    with pytest.raises(ReadonlyConfigError):
        del base.optim["momentum"]

    experiment = parser.compose_config(["solver=big", "lr=0.1"])
    assert not OmegaConf.is_readonly(experiment)
    experiment.optim.momentum = 0
    experiment["values"].append(3)
    raw = parser.compose(["solver=big"])
    raw["optim"]["momentum"] = -1
    assert base.optim.momentum == 0.9
    assert list(base["values"]) == [1, 2]
    assert parser.compose(["solver=big"])["optim"]["momentum"] == 0.9


def test_group_choices_are_distinct_and_raw_composition_shares_invalidation(root):
    parser = ConfigParser(root)
    small = parser.compose_base_config(["solver=small"])
    big = parser.compose_base_config(["solver=big"])
    assert (small.width, big.width) == (8, 16)
    # An unused config file does not evict or invalidate either base.
    write(root, "solver/unused", "ignored: true\n")
    assert parser.compose_base_config(["solver=small"]) is small
    assert parser.compose_base_config(["solver=big"]) is big
    # Invalidate via the ordinary dict path before requesting a cached base.
    write(root, "common", "# @package _global_\nshared: 12345\n")
    assert parser.compose(["solver=big"])["shared"] == 12345
    assert parser.compose_base_config(["solver=big"]).shared == 12345
    assert parser.compose_base_config(["solver=small"]).shared == 12345
    assert small.shared == big.shared == 1


def test_optional_file_creation_and_deletion_refresh_base(root):
    parser = ConfigParser(root)
    before = parser.compose_base_config()
    assert "extra" not in before
    write(root, "extra/absent", "value: 2\n")
    present = parser.compose_base_config()
    assert present.extra.value == 2
    (root / "extra/absent.yaml").unlink()
    absent = parser.compose_base_config()
    assert "extra" not in absent
    assert absent is not before and absent is not present
    # A required file disappearing must fail rather than return the stale base.
    (root / "common.yaml").unlink()
    with pytest.raises(ConfigError):
        parser.compose_base_config()


def test_clear_cache_lru_and_disabled_cache(root):
    parser = ConfigParser(root, cache_size=2)
    small = parser.compose_base_config(["solver=small"])
    big = parser.compose_base_config(["solver=big"])
    assert parser.compose_base_config(["solver=small"]) is small
    parser.compose_base_config()  # fills third slot, evicts big
    assert parser.compose_base_config(["solver=small"]) is small
    assert parser.compose_base_config(["solver=big"]) is not big
    assert len(parser._bases) == 2
    parser.clear_cache()
    assert parser.compose_base_config(["solver=small"]) is not small

    uncached = ConfigParser(root, cache_size=0)
    assert uncached.compose_base_config() is not uncached.compose_base_config()


def test_cached_interpolations_remain_lazy(root, monkeypatch):
    parser = ConfigParser(root)
    monkeypatch.setenv("DORA_PARSER_CACHE_TEST", "first")
    first = parser.compose_base_config()
    assert first.environment == "first"
    monkeypatch.setenv("DORA_PARSER_CACHE_TEST", "second")
    second = parser.compose_base_config()
    assert second is first
    assert second.environment == "second"

    name = "dora_parser_group_base_test"
    counter = iter(range(100))
    OmegaConf.register_new_resolver(name, lambda: next(counter), use_cache=True)
    try:
        write(root, "config", "value: ${" + name + ":}\n")
        first = parser.compose_base_config()
        assert first.value == first.value == 0
        second = parser.compose_base_config()
        assert second is first
        assert second.value == second.value == 1
    finally:
        OmegaConf.clear_resolver(name)


def test_explicit_immutable_mode_and_config_name(root):
    parser = ConfigParser(root, check_files=False)
    before = parser.compose_base_config()
    write(root, "common", "# @package _global_\nshared: 99\n")
    assert parser.compose_base_config() is before
    assert before.shared == 1
    parser.clear_cache()
    assert parser.compose_base_config().shared == 99

    write(root, "other", "value: other\n")
    parser.config_name = "other"
    assert parser.compose_base_config().value == "other"
    assert parser.compose() == {"value": "other"}
