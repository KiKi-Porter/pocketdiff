from pocketdiff.preprocessing import save_sample_cache, write_manifest
from pocketdiff.training.diffusion import DiffusionTrainConfig, load_apo2mol_slices


def test_diffusion_cache_manifest_supports_disjoint_train_valid_test(tmp_path):
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    source = tmp_path / "source.raw"
    source.write_text("source\n")
    entries = []
    for split, sample_id in (("train", "train-id"), ("valid", "valid-id"), ("test", "test-id")):
        value = _synthetic_complex()
        value.sample_id = sample_id
        entries.append(
            save_sample_cache(
                tmp_path / "cache" / f"{sample_id}.pt",
                value,
                source_paths={"source": source},
                split=split,
            )
        )
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, entries)
    config = DiffusionTrainConfig(
        cache_manifest=str(manifest),
        train_count=1,
        valid_count=1,
        test_count=1,
        holdout_count=0,
        updates=1,
    )
    slices = load_apo2mol_slices(config)
    assert slices["train"].sample_ids == ["train-id"]
    assert slices["valid"].sample_ids == ["valid-id"]
    assert slices["holdout"].sample_ids == ["valid-id"]
    assert slices["test"].sample_ids == ["test-id"]
    assert set(slices["train"].sample_ids).isdisjoint(slices["test"].sample_ids)
