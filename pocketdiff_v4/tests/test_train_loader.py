import pytest
from torch.utils.data import Dataset

import pocketdiff_v4.train as train


class _Ids(Dataset):
    def __init__(self, size):
        self.ids = list(range(size))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index]


def _ids_from_loader(loader):
    return [sample for batch in loader for sample in batch]


def test_single_rank_loader_keeps_final_partial_batch(monkeypatch):
    monkeypatch.setattr(train, "collate_complexes", lambda values: values)
    _, loader = train._build_train_loader(
        _Ids(10),
        batch_size=4,
        distributed=False,
        rank=0,
        world=1,
        num_workers=0,
        pin_memory=False,
    )

    values = _ids_from_loader(loader)

    assert len(loader) == 3
    assert sorted(values) == list(range(10))


def test_distributed_loaders_cover_every_sample_once(monkeypatch):
    monkeypatch.setattr(train, "collate_complexes", lambda values: values)
    all_values = []
    for rank in range(4):
        sampler, loader = train._build_train_loader(
            _Ids(12),
            batch_size=2,
            distributed=True,
            rank=rank,
            world=4,
            num_workers=0,
            pin_memory=False,
        )
        sampler.set_epoch(0)
        all_values.extend(_ids_from_loader(loader))

    assert sorted(all_values) == list(range(12))


def test_distributed_loader_rejects_uneven_dataset():
    with pytest.raises(ValueError, match="divisible by world size"):
        train._build_train_loader(
            _Ids(10),
            batch_size=2,
            distributed=True,
            rank=0,
            world=4,
            num_workers=0,
            pin_memory=False,
        )
