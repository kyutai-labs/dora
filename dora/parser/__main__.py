"""Compose and print a config without importing Hydra."""

import argparse
import json

import yaml  # type: ignore[import-untyped]

from . import ConfigError, ConfigParser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", default="config")
    parser.add_argument(
        "--resolve", action="store_true", help="Resolve interpolations with OmegaConf"
    )
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    try:
        composer = ConfigParser(args.config_dir, args.config_name)
        if args.resolve:
            from omegaconf import OmegaConf

            result = OmegaConf.to_container(composer.compose_config(args.overrides), resolve=True)
        else:
            result = composer.compose(args.overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(yaml.safe_dump(result, sort_keys=False), end="")


if __name__ == "__main__":
    main()
