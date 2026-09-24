from pocketdiff_v4.evaluate import _merge_split_rows


def test_merge_split_rows_supports_valid_only_shards():
    parts = [
        {"valid": [{"sample_id": "a"}]},
        {"valid": [{"sample_id": "b"}]},
    ]

    merged = _merge_split_rows(parts, ("valid",))

    assert [row["sample_id"] for row in merged["valid"]] == ["a", "b"]
