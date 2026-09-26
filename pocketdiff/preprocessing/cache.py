"""Versioned `.pt` cache for canonical PocketDiff samples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.training.clean import CleanExample, make_clean_example


CACHE_FORMAT = "pocketdiff-preprocessed-v1"
SCHEMA_VERSION = "schema-v1"
# v2 resolves late-bridge small rotations with atan2 in so3_log. Stored
# remaining-rotation labels from v1 must not silently enter new training.
GEOMETRY_VERSION = "geometry-v2"
ADAPTER_VERSION = "apo2mol-adapter-v1"


class CacheError(ValueError):
    """A cache is malformed, stale, or incompatible with the current schema."""


@dataclass(frozen=True)
class CachedSample:
    complex_value: PocketComplex
    clean_example: CleanExample
    metadata: Dict[str, object]


def _tensor_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [_tensor_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _tensor_to_cpu(item) for key, item in value.items()}
    return value


def _complex_payload(value: PocketComplex) -> Dict[str, object]:
    return {field.name: _tensor_to_cpu(getattr(value, field.name)) for field in fields(PocketComplex)}


def _complex_from_payload(payload: object) -> PocketComplex:
    if not isinstance(payload, dict):
        raise CacheError("cache complex payload must be a dictionary")
    expected = {field.name for field in fields(PocketComplex)}
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        raise CacheError(f"cache complex fields mismatch; missing={missing}, unexpected={unexpected}")
    try:
        return PocketComplex(**payload)
    except Exception as exc:
        raise CacheError(f"cached PocketComplex violates schema: {exc}") from exc


def fingerprint_file(path: Union[str, Path]) -> Dict[str, object]:
    file_path = Path(path)
    if not file_path.is_file():
        raise CacheError(f"source file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = file_path.stat()
    return {
        "path": str(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _verify_sources(source: Mapping[str, Mapping[str, object]]) -> None:
    for name, expected in source.items():
        current = fingerprint_file(expected["path"])
        if current["size"] != expected.get("size") or current["sha256"] != expected.get("sha256"):
            raise CacheError(
                f"source fingerprint mismatch for {name}: "
                f"expected sha256={expected.get('sha256')} size={expected.get('size')}, "
                f"got sha256={current['sha256']} size={current['size']}"
            )


def save_sample_cache(
    path: Union[str, Path],
    complex_value: PocketComplex,
    *,
    source_paths: Mapping[str, Union[str, Path]],
    split: str,
    adapter_version: str = ADAPTER_VERSION,
    geometry_version: str = GEOMETRY_VERSION,
) -> Dict[str, object]:
    """Serialize one canonical sample and return its manifest entry."""

    if not split:
        raise ValueError("split must be non-empty")
    source = {name: fingerprint_file(source_path) for name, source_path in source_paths.items()}
    clean = make_clean_example(complex_value)
    payload = {
        "format": CACHE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "geometry_version": geometry_version,
        "adapter_version": adapter_version,
        "sample_id": complex_value.sample_id,
        "split": split,
        "complex": _complex_payload(complex_value),
        "clean_target": {
            "target_translation_local": _tensor_to_cpu(clean.target_translation_local),
            "target_rotvec_local": _tensor_to_cpu(clean.target_rotvec_local),
            "target_valid": _tensor_to_cpu(clean.target_valid),
        },
        "source": source,
    }
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return {
        "sample_id": complex_value.sample_id,
        "split": split,
        "cache_path": str(cache_path),
        "schema_version": SCHEMA_VERSION,
        "geometry_version": geometry_version,
        "adapter_version": adapter_version,
        "source": source,
        "num_protein_atoms": complex_value.num_protein_atoms,
        "num_residues": complex_value.num_residues,
        "num_ligand_atoms": complex_value.num_ligand_atoms,
    }


def load_sample_cache(path: Union[str, Path], *, verify_sources: bool = False) -> CachedSample:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise CacheError(f"cache file does not exist: {cache_path}")
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        raise CacheError(f"cannot load cache file {cache_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise CacheError(f"unsupported cache format in {cache_path}")
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("geometry_version", GEOMETRY_VERSION),
        ("adapter_version", ADAPTER_VERSION),
    ):
        if payload.get(key) != expected:
            raise CacheError(f"{cache_path}: {key}={payload.get(key)!r}, expected {expected!r}")
    if verify_sources:
        source = payload.get("source")
        if not isinstance(source, dict):
            raise CacheError("cache source fingerprint is missing")
        _verify_sources(source)
    complex_value = _complex_from_payload(payload.get("complex"))
    if payload.get("sample_id") != complex_value.sample_id:
        raise CacheError("cache sample_id disagrees with PocketComplex")
    target = payload.get("clean_target")
    if not isinstance(target, dict):
        raise CacheError("cache clean_target is missing")
    try:
        clean = CleanExample(
            complex_value=complex_value,
            target_translation_local=target["target_translation_local"],
            target_rotvec_local=target["target_rotvec_local"],
            target_valid=target["target_valid"],
        )
    except Exception as exc:
        raise CacheError(f"cache clean_target is invalid: {exc}") from exc
    metadata = {
        "format": payload["format"],
        "schema_version": payload["schema_version"],
        "geometry_version": payload["geometry_version"],
        "adapter_version": payload["adapter_version"],
        "sample_id": payload["sample_id"],
        "split": payload.get("split"),
        "source": payload.get("source", {}),
    }
    return CachedSample(complex_value=complex_value, clean_example=clean, metadata=metadata)


def write_manifest(
    path: Union[str, Path],
    entries,
    *,
    config: Optional[Mapping[str, object]] = None,
    filtered_counts: Optional[Mapping[str, int]] = None,
) -> None:
    manifest_path = Path(path)
    payload = {
        "format": CACHE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "geometry_version": GEOMETRY_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "config": dict(config or {}),
        "filtered_counts": dict(filtered_counts or {}),
        "entries": list(entries),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_manifest(path: Union[str, Path]) -> Dict[str, object]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise CacheError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise CacheError("unsupported or malformed cache manifest")
    for key, expected in (("schema_version", SCHEMA_VERSION), ("geometry_version", GEOMETRY_VERSION), ("adapter_version", ADAPTER_VERSION)):
        if payload.get(key) != expected:
            raise CacheError(f"manifest {key} mismatch: {payload.get(key)!r} != {expected!r}")
    if not isinstance(payload.get("entries"), list):
        raise CacheError("manifest entries must be a list")
    return payload


def load_cached_clean_examples(
    manifest_path: Union[str, Path],
    *,
    verify_sources: bool = True,
):
    """Load clean examples in manifest order with optional source verification."""

    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    examples = []
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or "cache_path" not in entry:
            raise CacheError("manifest entry is missing cache_path")
        cache_path = Path(entry["cache_path"])
        if not cache_path.is_absolute():
            local_candidate = manifest_file.parent / cache_path
            cache_path = local_candidate if local_candidate.is_file() else cache_path
        cached = load_sample_cache(cache_path, verify_sources=verify_sources)
        if cached.complex_value.sample_id != entry.get("sample_id"):
            raise CacheError(
                f"manifest/cache sample identity mismatch: {entry.get('sample_id')} != {cached.complex_value.sample_id}"
            )
        examples.append(cached.clean_example)
    return examples


__all__ = [
    "ADAPTER_VERSION",
    "CACHE_FORMAT",
    "CacheError",
    "CachedSample",
    "GEOMETRY_VERSION",
    "SCHEMA_VERSION",
    "fingerprint_file",
    "load_manifest",
    "load_cached_clean_examples",
    "load_sample_cache",
    "save_sample_cache",
    "write_manifest",
]
