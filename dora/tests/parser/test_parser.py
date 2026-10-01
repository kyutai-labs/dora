import math
import subprocess
import sys

import pytest
from hydra import compose as hydra_compose
from hydra import initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from dora.parser import ConfigError, ConfigParser, UnsupportedFeature, parse_override, parse_value


def write(root, name, text):
    path = root / (name + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def oracle(root, overrides=()):
    with initialize_config_dir(str(root), version_base="1.1"):
        return hydra_compose("config", list(overrides))


def assert_same(left, right):
    assert type(left) is type(right), (left, right)
    if isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


@pytest.mark.parametrize(
    "text",
    [
        "",
        "null",
        "NuLl",
        "true",
        "False",
        "yes",
        "off",
        "01",
        "0x10",
        "1_000",
        "1e-5",
        "-1e6",
        "+.5",
        "1.",
        "-inf",
        "NaN",
        "a b",
        "//reference/foo",
        "'true'",
        '"01"',
        r'"a\nb"',
        r'"\u00e9"',
        "'café d’essai'",
        r'"a\\b"',
        r"'a\'b'",
        r"a\,b",
        r"a\=b",
        r"a\ b",
        r"a\:b",
        r"C:\foo",
        r"\ leading\ ",
        r"trailing\  ",
        r'"trailing\\"',
        r'"middle\\\"quote"',
        "${optim.lr}",
        "${oc.env:USER,anonymous}",
        "prefix-${seed}",
        "[]",
        "{}",
        "[1, 2e-5, null, false, yes, 'a,b', [3]]",
        "{optim.lr:1e-5,a:[true,{b:foo}],str:01}",
        "{null:foo,True:bar,10:baz}",
        "[${oc.env:USER,anonymous},foo]",
    ],
)
def test_cli_values_match_hydra(text):
    expected = OverridesParser.create().parse_override("x=" + text).value()
    assert_same(parse_value(text), expected)


@pytest.mark.parametrize(
    "text",
    [
        "x=1,2",
        "x=range(1,4)",
        "x=[1,]",
        "x={a:}",
        "x='oops",
        "x=[1",
        "x=[1]junk",
        "x",
        "=1",
    ],
)
def test_invalid_and_unsupported_overrides(text):
    with pytest.raises(ConfigError):
        parse_override(text)


@pytest.fixture
def config(tmp_path):
    write(
        tmp_path,
        "config",
        """defaults:
  - _self_
  - data: first
  - solver: default
  - optional extra: absent
x: 1
nullable: null
missing: ???
mapping: {a: 1, b: 2}
items: [1, 2, 3]
alias: ${mapping}
interpolated: ${x}
scientific: 1e-5
date: 2026-09-09
""",
    )
    write(tmp_path, "data/first", "# @package _global_\ndata: first\n")
    write(tmp_path, "data/second", "# @package __global__\ndata: second\n")
    write(tmp_path, "solver/default", "# @package _global_\nsolver: base\n")
    write(
        tmp_path,
        "solver/deep/main",
        """# @package _global_
defaults:
  - /solver/default
  - /model: default
  - override /data: second
  - override /scale: small
  - _self_
solver: deep
x: 2
""",
    )
    write(tmp_path, "model/default", "# @package _global_\ndefaults:\n  - /scale: big\nwidth: 20\n")
    write(tmp_path, "scale/big", "# @package _global_\nwidth: 100\n")
    write(tmp_path, "scale/small", "# @package _global_\nwidth: 10\n")
    return tmp_path


@pytest.mark.parametrize(
    "overrides",
    [
        [],
        ["solver=deep/main"],
        ["solver=deep/main", "data=first", "scale=big"],
        ["x=4", "interpolated=8"],
        ["+new.path=5"],
        ["++mapping.a=3"],
        ["mapping={a:9}"],
        ["+mapping={c:3}"],
        ["mapping=null"],
        ["+nullable=4"],
        ["+missing=4"],
        ["items=[4]"],
        ["items.1=9"],
        ["~mapping.a=1"],
        ["~x"],
        ["~data"],
        ["~data=first"],
        ["+scale=small"],
        ["x=3", "x=4"],
        ["mapping.a=???"],
        ["+nullable.deep=true"],
    ],
)
def test_composition_matches_hydra(config, overrides):
    parser = ConfigParser(config)
    expected = oracle(config, overrides)
    assert_same(parser.compose(overrides), OmegaConf.to_container(expected, resolve=False))
    try:
        resolved = OmegaConf.to_container(expected, resolve=True)
    except Exception as exc:  # noqa: BLE001 -- match the oracle exception type
        with pytest.raises(type(exc)):
            OmegaConf.to_container(parser.compose_config(overrides), resolve=True)
    else:
        assert_same(
            OmegaConf.to_container(parser.compose_config(overrides), resolve=True), resolved
        )


@pytest.mark.parametrize(
    "overrides",
    [
        ["typo=4"],
        ["mapping={typo:4}"],
        ["+x=4"],
        ["~nullable"],
        ["~x=42"],
        ["scale=small"],
        ["data=absent"],
        ["~data=second"],
        ["items.4=1"],
        ["solver=deep/main", "+scale=small"],
        ["data=null"],
    ],
)
def test_rejects_invalid_compositions_like_hydra(config, overrides):
    # Invalid config-group value types raise plain ValueError in Hydra.
    with pytest.raises((HydraException, ValueError)):
        oracle(config, overrides)
    with pytest.raises(ConfigError):
        ConfigParser(config).compose(overrides)


def test_caches_are_independent_and_refresh(config):
    parser = ConfigParser(config)
    first = parser.compose(["solver=deep/main"])
    first["mapping"]["a"] = 999
    first["items"].append(999)
    assert parser.compose(["solver=deep/main"])["mapping"]["a"] == 1
    assert parser.compose(["solver=deep/main"])["items"] == [1, 2, 3]
    write(config, "scale/small", "# @package _global_\nwidth: 12345\n")
    # small is overwritten by model's implicit _self_, so change a winning file.
    write(config, "solver/default", "# @package _global_\nsolver: changed\nnew_value: 12345\n")
    assert parser.compose(["solver=deep/main"])["new_value"] == 12345
    write(config, "extra/absent", "enabled: true\n")
    assert parser.compose()["extra"]["enabled"] is True
    parser.clear_cache()
    assert parser.compose()["solver"] == "changed"


@pytest.mark.parametrize("header", ["", "# @package custom\n", "# @package _global_\n"])
@pytest.mark.parametrize("relocation", ["", "@relocated", "@_global_", "@_here_"])
def test_package_rules(tmp_path, header, relocation):
    write(tmp_path, "config", "defaults:\n  - parent: default\n")
    write(tmp_path, "parent/default", f"defaults:\n  - child{relocation}: first\nouter: 1\n")
    write(tmp_path, "parent/child/first", header + "value: 2\n")
    assert_same(
        ConfigParser(tmp_path).compose(), OmegaConf.to_container(oracle(tmp_path), resolve=False)
    )


def test_relocated_group_override(tmp_path):
    write(tmp_path, "config", "defaults:\n  - db@src: a\n  - db@dst: a\n")
    write(tmp_path, "db/a", "port: 1\n")
    write(tmp_path, "db/b", "port: 2\n")
    args = ["db@src=b"]
    assert_same(
        ConfigParser(tmp_path).compose(args), OmegaConf.to_container(oracle(tmp_path, args))
    )


def test_self_order_and_missing_merge(tmp_path):
    write(tmp_path, "config", "defaults: [base, _self_]\nx: ???\nnested: {x: '???'}\n")
    write(tmp_path, "base", "x: 3\nnested: {x: 4}\n")
    assert_same(ConfigParser(tmp_path).compose(), OmegaConf.to_container(oracle(tmp_path)))


@pytest.mark.parametrize(
    "text",
    [
        "defaults: [config]",
        "defaults: [_self_, _self_]",
        "x: 1\nx: 2",
        "defaults: [{unknown: ???}]",
        "defaults: [{data: '${other}'}]",
        "hydra: {searchpath: [pkg://elsewhere]}",
        "defaults: [{data: [a, b]}]",
    ],
)
def test_bad_configs_are_explicit(tmp_path, text):
    write(tmp_path, "config", text)
    with pytest.raises(ConfigError):
        ConfigParser(tmp_path).compose()


def test_no_hydra_import_and_no_global_loader_mutation():
    script = """
import sys, yaml
before = yaml.safe_load('x: 1e-5\\ndate: 2026-09-09')
import dora.parser
assert 'hydra' not in sys.modules
assert 'omegaconf' not in sys.modules
assert yaml.safe_load('x: 1e-5\\ndate: 2026-09-09') == before
"""
    subprocess.run([sys.executable, "-B", "-c", script], check=True)


def test_lazy_interpolation_and_struct_mode(config):
    cfg = ConfigParser(config).compose_config()
    cfg.x = 7
    assert cfg.interpolated == 7
    with pytest.raises(ConfigAttributeError):
        cfg.typo = 1
    with pytest.raises(UnsupportedFeature):
        ConfigParser(config).compose(["alias.a=5"])


def test_config_paths_stay_under_root(config):
    with pytest.raises(ConfigError):
        ConfigParser(config, "../config").compose()


def test_relative_defaults_are_relative_to_selected_group(tmp_path):
    # A slash in an option name does not move the group's defaults search root.
    write(tmp_path, "config", "defaults: [{solver: 'arflow/debug'}]\n")
    write(tmp_path, "solver/arflow/debug", "defaults: [arflow/base]\ndebug: true\n")
    write(tmp_path, "solver/arflow/base", "defaults: [leaf]\nvalue: 2\n")
    write(tmp_path, "solver/arflow/leaf", "leaf: 3\n")
    args = []
    assert_same(
        ConfigParser(tmp_path).compose(args), OmegaConf.to_container(oracle(tmp_path, args))
    )


def test_relative_parent_include_within_config_root(tmp_path):
    write(tmp_path, "config", "defaults: [{solver: nested}]\n")
    write(tmp_path, "solver/nested", "# @package _global_\ndefaults: [../base]\n")
    write(tmp_path, "base", "# @package _global_\nvalue: 1\n")
    assert_same(ConfigParser(tmp_path).compose(), OmegaConf.to_container(oracle(tmp_path)))
