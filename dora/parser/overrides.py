"""Small recursive descent parser for Hydra's single-run override syntax.

This deliberately does not use YAML for CLI values: Hydra treats ``yes``,
``01`` and quoted escapes differently from YAML (and from JSON).
"""

import re
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError, UnsupportedFeature

_UINT = r"(?:0|[1-9](?:_?[0-9])*)"
_INT = re.compile(rf"[+-]?{_UINT}\Z")
_POINT = rf"(?:{_UINT}\.|(?:{_UINT})?\.[0-9](?:_?[0-9])*)"
_FLOAT = re.compile(
    rf"[+-]?(?:{_POINT}|(?:{_UINT}|{_POINT})[eE][+-]?[0-9](?:_?[0-9])*|inf|nan)\Z",
    re.IGNORECASE,
)
_KEY = re.compile(r"[\w$-]+(?:[./][\w$-]+)*(?:@[\w$.-]*)?\Z", re.ASCII)
_ESCAPABLE = set("\\()[]{}:=, \t")


def _scalar(text: str) -> Any:
    lower = text.lower()
    if lower == "null":
        return None
    if lower in ("true", "false"):
        return lower == "true"
    if _INT.fullmatch(text):
        return int(text.replace("_", ""))
    if _FLOAT.fullmatch(text):
        return float(text.replace("_", ""))
    return text


class _Reader:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def error(self, message):
        raise ConfigError(f"{message} at column {self.pos + 1} in {self.text!r}")

    def quoted(self):
        quote = self.text[self.pos]
        self.pos += 1
        out = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            self.pos += 1
            if ch == quote:
                return "".join(out)
            if ch == "\\":
                start = self.pos - 1
                while self.pos < len(self.text) and self.text[self.pos] == "\\":
                    self.pos += 1
                count = self.pos - start
                if self.pos < len(self.text) and self.text[self.pos] == quote:
                    out.append("\\" * (count // 2))
                    self.pos += 1
                    if count % 2 == 0:
                        return "".join(out)
                    out.append(quote)
                else:
                    out.append("\\" * count)
                continue
            out.append(ch)
        self.error("Unterminated quoted value")

    def primitive(self, *, key=False):
        out = []
        escaped = False
        trailing_space = 0
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch in ",]}" or (key and ch == ":"):
                break
            if not key and self.text.startswith("${", self.pos):
                # Interpolations are kept verbatim for OmegaConf at the boundary.
                start = self.pos
                end = self.text.find("}", start + 2)
                if end < 0:
                    self.error("Unterminated interpolation")
                self.pos = end + 1
                out.append(self.text[start : self.pos])
                trailing_space = 0
                continue
            if ch == "\\" and self.pos + 1 < len(self.text):
                following = self.text[self.pos + 1]
                if following in _ESCAPABLE:
                    out.append(following)
                    escaped = True
                    trailing_space = 0
                    self.pos += 2
                    continue
            if ch in "()":
                raise UnsupportedFeature(
                    "Override functions/sweeps are not supported; quote literal parentheses"
                )
            if ch in "[{'\"}=\n\r" or (key and ch == "$"):
                self.error("Unexpected character (quote or escape literal punctuation)")
            if not (ch.isascii() and (ch.isalnum() or ch in "_-/\\+.$%*@?|: \t")):
                self.error("Unquoted character is outside Hydra's grammar")
            out.append(ch)
            trailing_space = trailing_space + 1 if ch in " \t" else 0
            self.pos += 1
        result = "".join(out)
        if trailing_space:
            result = result[:-trailing_space]
        if not result:
            self.error("Expected a value")
        return result if escaped else _scalar(result)

    def element(self):
        self.ws()
        if self.pos == len(self.text):
            self.error("Expected a value")
        ch = self.text[self.pos]
        if ch in "\"'":
            return self.quoted()
        if ch not in "[{":
            return self.primitive()
        self.pos += 1
        close = "]" if ch == "[" else "}"
        result: Any = [] if ch == "[" else {}
        self.ws()
        if self.pos < len(self.text) and self.text[self.pos] == close:
            self.pos += 1
            return result
        while True:
            if ch == "[":
                result.append(self.element())
            else:
                key = self.primitive(key=True)
                self.ws()
                if self.pos >= len(self.text) or self.text[self.pos] != ":":
                    self.error("Expected ':' after dictionary key")
                self.pos += 1
                result[key] = self.element()
            self.ws()
            if self.pos >= len(self.text):
                self.error(f"Expected {close!r}")
            sep = self.text[self.pos]
            self.pos += 1
            if sep == close:
                return result
            if sep != ",":
                self.error(f"Expected ',' or {close!r}")
            self.ws()


def parse_value(text: str) -> Any:
    """Parse one CLI value, preserving interpolation and Hydra string escapes."""
    if not text.strip():
        return ""
    reader = _Reader(text)
    result = reader.element()
    reader.ws()
    if reader.pos != len(text):
        if text[reader.pos] == ",":
            raise UnsupportedFeature(
                "Choice sweeps are not supported; use a list or quote the comma"
            )
        reader.error("Unexpected trailing input")
    return result


@dataclass(frozen=True)
class Override:
    key: str
    value: Any
    operation: str = "set"
    has_value: bool = True


def parse_override(text: str) -> Override:
    """Parse ``key=value``, ``+key=value``, ``++key=value`` or ``~key[=value]``."""
    if not isinstance(text, str):
        raise ConfigError("Overrides must be strings")
    operation = "set"
    for prefix, op in (("++", "force"), ("+", "add"), ("~", "delete")):
        if text.startswith(prefix):
            text, operation = text[len(prefix) :], op
            break
    key, sep, value = text.partition("=")
    if not _KEY.fullmatch(key):
        raise ConfigError(f"Invalid override key: {key!r}")
    if not sep and operation != "delete":
        raise ConfigError(f"Override requires '=': {text!r}")
    return Override(key, parse_value(value) if sep else None, operation, bool(sep))
