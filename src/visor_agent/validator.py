"""[SPEC-004] DAG validator.

Per node:
1. JSON Schema validation of `params` against the skill's `input_schema`.
2. Path validation: `input_path` / `output_path` must match `absolute-storage-path`
   format (prefix = `/visor/`) and contain no shell metacharacters.
3. Composite skill nodes MUST have been resolved to atomic skills upstream.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import jsonschema


STORAGE_PREFIX = "/visor/"
SHELL_META_RE = re.compile(r"[;|`$&]|&&|\|\||\$\(")
PATH_SUFFIXES = ("_path", "_zarr")


def _validate_path(value: str) -> None:
    if not value.startswith(STORAGE_PREFIX):
        raise ValueError(f"path must be under {STORAGE_PREFIX}: {value!r}")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"path contains '..' segment: {value!r}")
    if SHELL_META_RE.search(value):
        raise ValueError(f"path contains shell metacharacters: {value!r}")


def _format_path_keys(schema: dict[str, Any]) -> set[str]:
    properties = schema.get("properties", {})
    return {
        key
        for key, spec in properties.items()
        if isinstance(spec, dict) and spec.get("format") == "absolute-storage-path"
    }


def validate_node(
    params: dict[str, Any], skill_spec: dict[str, Any]
) -> None:
    schema = skill_spec["interface"]["input_schema"]
    jsonschema.validate(params, schema)
    path_keys = _format_path_keys(schema)
    for key, value in params.items():
        if isinstance(value, str) and (key in path_keys or key.endswith(PATH_SUFFIXES)):
            _validate_path(params[key])


def validate_dag(dag: dict[str, Any], registry: list[dict[str, Any]]) -> None:
    by_id = {s["id"]: s for s in registry}
    for node in dag["skills"]:
        spec = by_id.get(node["id"])
        if spec is None:
            raise ValueError(f"unknown skill in DAG: {node['id']}")
        if spec["type"] != "atomic":
            raise ValueError(f"DAG must contain only atomic skills: {node['id']}")
        validate_node(node["params"], spec)
