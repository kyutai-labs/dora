class ConfigError(ValueError):
    """Invalid configuration or override."""


class UnsupportedFeature(ConfigError):
    """Hydra feature outside this experimental parser's supported subset."""
