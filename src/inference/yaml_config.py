"""Strict YAML mapping input for the private product runtime."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import yaml

_INTERPOLATION = re.compile(r"\$\{")


class YamlConfigError(ValueError):
    """Raised when a product YAML document is not a literal mapping."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, Hashable):
            raise YamlConfigError("YAML mapping keys must be hashable")
        if key in mapping:
            raise YamlConfigError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _reject_interpolation(value: Any, location: str = "root") -> None:
    if isinstance(value, str):
        if _INTERPOLATION.search(value):
            raise YamlConfigError(f"unresolved interpolation at {location}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_interpolation(key, f"{location}.<key>")
            _reject_interpolation(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_interpolation(child, f"{location}[{index}]")


def load_yaml_mapping(path: Path) -> dict[Any, Any]:
    """Load a literal YAML mapping without OmegaConf resolution semantics."""

    try:
        document = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except yaml.YAMLError as exc:
        raise YamlConfigError(f"invalid YAML document: {path}") from exc
    if not isinstance(document, Mapping):
        raise YamlConfigError(f"YAML document root must be a mapping: {path}")
    _reject_interpolation(document)
    return dict(document)
