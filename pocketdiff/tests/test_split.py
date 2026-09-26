import pytest

from pocketdiff.tests.test_training import _complex
from pocketdiff.training import deterministic_clean_split, make_clean_example


def _examples(count=6):
    return [
        make_clean_example(_complex(f"sample-{index}", 0.05 * index, [0.1 * index, 0.0, 0.0]))
        for index in range(count)
    ]


def test_deterministic_clean_split_is_reproducible_and_disjoint():
    examples = _examples()
    first = deterministic_clean_split(examples, holdout=2, seed=17)
    second = deterministic_clean_split(examples, holdout=2, seed=17)
    assert first.permutation == second.permutation
    assert first.train_ids == second.train_ids
    assert first.holdout_ids == second.holdout_ids
    assert set(first.train_ids).isdisjoint(first.holdout_ids)
    assert set(first.train_ids + first.holdout_ids) == {f"sample-{index}" for index in range(6)}


@pytest.mark.parametrize("holdout", [0, 6, 7])
def test_deterministic_clean_split_rejects_invalid_holdout(holdout):
    with pytest.raises(ValueError, match="holdout"):
        deterministic_clean_split(_examples(), holdout=holdout, seed=1)
