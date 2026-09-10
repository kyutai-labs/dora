"""Safe YAML with OmegaConf's scalar conventions, using libyaml if present."""

import re

import yaml  # type: ignore[import-untyped]


class ConfigLoader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):  # type: ignore[misc]
    def construct_mapping(self, node, deep=False):
        # Check before YAML merge-key expansion, like OmegaConf's loader.
        keys = set()
        for key, _ in node.value:
            if key.tag == "tag:yaml.org,2002:str":
                if key.value in keys:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"Duplicate key {key.value!r}", key.start_mark
                    )
                keys.add(key.value)
        return super().construct_mapping(node, deep=deep)


# Keep timestamps as strings; add scientific notation without a decimal point.
# Resolver tables belong to this loader; never change global PyYAML behavior.
ConfigLoader.yaml_implicit_resolvers = {
    key: [(tag, rx) for tag, rx in rules if tag != "tag:yaml.org,2002:timestamp"]
    for key, rules in ConfigLoader.yaml_implicit_resolvers.items()
}
ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[+-]?[0-9]+(?:_[0-9]+)*(?:\.[0-9_]*)?[eE][+-]?[0-9]+$"),
    list("-+0123456789"),
)


def load_yaml(text):
    return yaml.load(text, Loader=ConfigLoader)
