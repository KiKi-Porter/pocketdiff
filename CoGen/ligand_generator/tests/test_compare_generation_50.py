import torch
import numpy as np

from scripts.compare_generation_50 import (
    _paired_bootstrap,
    _pair_distance_histograms,
    evaluate_generated,
    protein_only,
    select_test_indices,
)
import utils.reconstruct as reconstruct


def _records_for_trajectories(counts):
    records = []
    sample_index = 0
    test_indices = []
    for trajectory_index, count in enumerate(counts):
        trajectory_id = f"trajectory_{trajectory_index}"
        for cluster_id in range(count):
            records.append(
                {
                    "sample_index": sample_index,
                    "trajectory_id": trajectory_id,
                    "split": "test",
                    "cluster_id": cluster_id,
                    "cluster_frame": cluster_id * 10,
                }
            )
            test_indices.append(sample_index)
            sample_index += 1
    return records, test_indices


def test_selection_spreads_samples_and_caps_each_trajectory_at_two():
    records, test_indices = _records_for_trajectories([5] * 30)

    selected = select_test_indices(records, test_indices, num_samples=50, seed=17)
    selected_trajectories = [
        next(
            record["trajectory_id"]
            for record in records
            if record["sample_index"] == index
        )
        for index in selected
    ]

    assert len(selected) == 50
    assert len(set(selected)) == 50
    assert len(set(selected_trajectories)) == 30
    assert max(
        selected_trajectories.count(value)
        for value in set(selected_trajectories)
    ) == 2
    assert select_test_indices(records, test_indices, 50, 17) == selected


def test_protein_only_does_not_depend_on_reference_ligand_fields():
    sample = {
        "protein_pos": torch.zeros((2, 3)),
        "protein_atom_feature": torch.zeros((2, 27)),
        "protein_element": torch.tensor([6, 7]),
        "ligand_pos": torch.randn((100, 3)),
        "ligand_element": torch.full((100,), 6),
        "ligand_bond_index": torch.zeros((2, 0), dtype=torch.long),
    }

    protein_pos, protein_features, protein_elements = protein_only(sample)

    assert tuple(protein_pos.shape) == (2, 3)
    assert tuple(protein_features.shape) == (2, 27)
    assert protein_elements.tolist() == [6, 7]


def test_paired_bootstrap_is_reproducible_and_reports_new_minus_old():
    old = [1.0, 2.0, 3.0]
    new = [1.5, 2.5, 3.5]

    first = _paired_bootstrap(old, new, seed=123, num_bootstrap=200)
    second = _paired_bootstrap(old, new, seed=123, num_bootstrap=200)

    assert first == second
    assert first["n_pairs"] == 3
    assert first["mean_new_minus_old"] == 0.5
    assert first["ci95"] == [0.5, 0.5]


def test_paired_bootstrap_resamples_trajectory_clusters_and_applies_mask():
    old = [0.0, 0.0, 0.0, 0.0, 0.0]
    new = [1.0, 1.0, 3.0, 3.0, 100.0]
    clusters = ["traj_a", "traj_a", "traj_b", "traj_b", "traj_c"]
    include = [True, True, True, True, False]

    result = _paired_bootstrap(
        old,
        new,
        seed=123,
        num_bootstrap=500,
        clusters=clusters,
        include=include,
    )
    repeated = _paired_bootstrap(
        old,
        new,
        seed=123,
        num_bootstrap=500,
        clusters=clusters,
        include=include,
    )

    assert result == repeated
    assert result["n_pairs"] == 4
    assert result["n_clusters"] == 2
    assert result["bootstrap_unit"] == "trajectory"
    assert result["mean_new_minus_old"] == 2.0
    assert result["ci95"][0] >= 1.0
    assert result["ci95"][1] <= 3.0


def test_pair_distance_histogram_is_compact_and_counts_pairs():
    histogram = _pair_distance_histograms(
        np.asarray([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32),
        [6, 6],
    )

    assert len(histogram["CC_2A"]) == 101
    assert len(histogram["All_12A"]) == 101
    assert sum(histogram["CC_2A"]) == 1
    assert sum(histogram["All_12A"]) == 1


def test_reconstruction_receives_float64_coordinates(monkeypatch):
    observed = {}

    def fake_reconstruct(pos, atom_types, aromatic=None, basic_mode=True):
        observed["dtype"] = pos.dtype
        raise reconstruct.MolReconsError()

    monkeypatch.setattr(reconstruct, "reconstruct_from_generated", fake_reconstruct)
    evaluate_generated(
        ligand_pos=np.asarray([[0, 0, 0], [1.5, 0, 0]], dtype=np.float32),
        ligand_classes=np.asarray([1, 1], dtype=np.int64),
        protein_pos=torch.tensor([[100.0, 0.0, 0.0]], dtype=torch.float32),
        protein_elements=torch.tensor([6], dtype=torch.long),
    )

    assert observed["dtype"] == np.float64
