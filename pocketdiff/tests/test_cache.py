from pathlib import Path

import pytest
import torch

from pocketdiff.preprocessing import (
    CacheError,
    load_manifest,
    load_sample_cache,
    save_sample_cache,
    write_manifest,
)
from pocketdiff.tests.test_training import _complex
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import collate_clean_examples


def test_sample_cache_round_trip_and_manifest(tmp_path):
    source_paths = {}
    for name in ("holo", "apo", "ligand"):
        path = tmp_path / f"{name}.raw"
        path.write_text(name + "\n")
        source_paths[name] = path
    cache_path = tmp_path / "cache" / "sample.pt"
    value = _complex("cache-sample", 0.2, [0.2, 0.1, 0.0])
    entry = save_sample_cache(cache_path, value, source_paths=source_paths, split="train")
    loaded = load_sample_cache(cache_path, verify_sources=True)
    assert loaded.complex_value.sample_id == value.sample_id
    assert torch.equal(loaded.complex_value.protein_pos_holo, value.protein_pos_holo)
    assert torch.equal(loaded.complex_value.ligand_type_ref, value.ligand_type_ref)
    assert torch.equal(loaded.clean_example.target_translation_local, loaded.clean_example.target_translation_local)
    assert entry["schema_version"] == "schema-v1"

    batch = collate_clean_examples([loaded.clean_example])
    prediction = PocketDiffModel(encoder_layers=1, knn=4)(**batch.model_kwargs())
    assert prediction.remaining_translation_local.shape == (1, 3)

    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, [entry], config={"limit": 1})
    manifest = load_manifest(manifest_path)
    assert manifest["entries"][0]["sample_id"] == "cache-sample"
    assert manifest["config"] == {"limit": 1}


def test_cache_source_change_is_rejected(tmp_path):
    source = tmp_path / "source.raw"
    source.write_text("before\n")
    cache_path = tmp_path / "sample.pt"
    save_sample_cache(
        cache_path,
        _complex("stale", -0.1, [0.1, 0.0, 0.0]),
        source_paths={"source": source},
        split="train",
    )
    source.write_text("after\n")
    with pytest.raises(CacheError, match="fingerprint mismatch"):
        load_sample_cache(cache_path, verify_sources=True)


def test_cache_format_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source.raw"
    source.write_text("source\n")
    cache_path = tmp_path / "sample.pt"
    save_sample_cache(
        cache_path,
        _complex("format", 0.0, [0.0, 0.0, 0.0]),
        source_paths={"source": source},
        split="train",
    )
    payload = torch.load(cache_path, map_location="cpu")
    payload["format"] = "old-format"
    torch.save(payload, cache_path)
    with pytest.raises(CacheError, match="unsupported cache format"):
        load_sample_cache(cache_path)


def test_old_geometry_labels_and_manifest_are_rejected(tmp_path):
    source = tmp_path / "source.raw"
    source.write_text("source\n")
    cache_path = tmp_path / "sample.pt"
    entry = save_sample_cache(
        cache_path, _complex("old-geometry", 7e-4, [0.1, 0., 0.]),
        source_paths={"source": source}, split="train", geometry_version="geometry-v1",
    )
    with pytest.raises(CacheError, match="geometry_version="):
        load_sample_cache(cache_path)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, [entry], config={})
    import json
    manifest = json.loads(manifest_path.read_text())
    manifest["geometry_version"] = "geometry-v1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CacheError, match="geometry_version mismatch"):
        load_manifest(manifest_path)
