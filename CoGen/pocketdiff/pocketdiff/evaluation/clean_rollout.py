"""Read-only scoring after autonomous inference has finished."""

import torch

from pocketdiff.geometry.frames import build_residue_frames


@torch.no_grad()
def score_clean_rollout(trace, inputs, holo_pos, sample_ids):
    """Use a fixed apo-valid atom set, even when later frames become invalid.

    All coordinates are compared in the original apo frame without fitting.
    Aggregation is an equal mean over graphs, not over atoms or time.
    """
    apo = inputs['apo_pos_ref']
    if holo_pos.shape != apo.shape or not torch.isfinite(holo_pos).all():
        raise ValueError('holo_pos must match apo shape and be finite')
    residue_count = inputs['residue_type'].numel()
    if (trace.positions.shape != (21,) + tuple(apo.shape)
            or trace.frame_valid.shape != (21, residue_count)
            or trace.update_valid.shape != (20, residue_count)
            or not torch.equal(trace.positions[0], apo)):
        raise ValueError('trace must contain apo followed by 20 states and corresponding residue masks')
    if not torch.isfinite(trace.positions).all():
        raise ValueError('trace coordinates must be finite')
    graph_count = int(inputs['batch_protein'].max()) + 1
    if len(sample_ids) != graph_count or len(set(sample_ids)) != graph_count:
        raise ValueError('sample_ids must identify every graph uniquely')
    apo_frames = build_residue_frames(apo, inputs['atom_to_residue_global'], inputs['protein_atom_name'],
                                      num_residues=residue_count)
    atom_valid = apo_frames.valid[inputs['atom_to_residue_global']]
    backbone = torch.tensor([n in ('N', 'CA', 'C', 'O') for n in inputs['protein_atom_name']],
                            dtype=torch.bool, device=apo.device)
    rows = []
    for step, pos in enumerate(trace.positions):
        for graph, sample_id in enumerate(sample_ids):
            atoms = atom_valid & (inputs['batch_protein'] == graph)
            bb = atoms & backbone
            residues = inputs['batch_residue'] == graph
            if not bool(atoms.any()) or not bool(bb.any()):
                raise ValueError('no apo-valid atoms/backbone for ' + sample_id)
            def rmsd(a, b, mask):
                return float((a[mask]-b[mask]).square().sum(-1).mean().sqrt())
            prev = trace.positions[max(0, step-1)]
            cumulative = torch.linalg.vector_norm(pos[atoms]-apo[atoms], dim=-1)
            increment = torch.linalg.vector_norm(pos[atoms]-prev[atoms], dim=-1)
            rows.append(dict(sample_id=sample_id, step=step,
                             holo_rmsd=rmsd(pos, holo_pos, atoms),
                             backbone_holo_rmsd=rmsd(pos, holo_pos, bb),
                             apo_displacement_rmsd=rmsd(pos, apo, atoms),
                             max_apo_displacement=float(cumulative.max()),
                             step_displacement_rmsd=rmsd(pos, prev, atoms),
                             max_step_displacement=float(increment.max()),
                             invalid_frame_count=int((~trace.frame_valid[step, residues]).sum()),
                             lost_apo_frame_count=int((apo_frames.valid[residues] &
                                                      ~trace.frame_valid[step, residues]).sum()),
                             skipped_valid_update_count=(int((apo_frames.valid[residues] &
                                                              ~trace.update_valid[step-1, residues]).sum())
                                                         if step else 0),
                             scored_atom_count=int(atoms.sum()), scored_backbone_count=int(bb.sum())))
    keys = ('holo_rmsd', 'backbone_holo_rmsd', 'apo_displacement_rmsd', 'step_displacement_rmsd')
    per_step = []
    for step in range(21):
        subset = [r for r in rows if r['step'] == step]
        aggregate = {key: sum(row[key] for row in subset)/graph_count for key in keys}
        for key in ('max_apo_displacement', 'max_step_displacement'):
            aggregate[key] = max(row[key] for row in subset)
        for key in ('invalid_frame_count', 'lost_apo_frame_count', 'skipped_valid_update_count'):
            aggregate[key] = sum(row[key] for row in subset)
        per_step.append(dict(step=step, **aggregate))
    return dict(mode='autonomous_clean', reduction='mean_over_graphs_at_each_step',
                scoring_mask='fixed_apo_valid_atoms', alignment='none_after_preprocessing',
                per_step=per_step, per_graph=rows)


__all__ = ['score_clean_rollout']
