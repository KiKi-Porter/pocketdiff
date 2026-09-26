"""Read-only TargetDiff UniTransformer/O(2) encoder adapter.

This module is intentionally separate from PocketDiffModel until its contract
is validated. It imports the official implementation lazily and never mutates
its source tree or coordinates (``fix_x=True``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


TARGETDIFF_ROOT = Path(__file__).resolve().parents[2] / 'targetdiff-main' / 'targetdiff-main'
TARGETDIFF_ENCODER_CONTRACT = 'targetdiff-uni-o2-readonly-v1'


class TargetDiffEncoderAdapter(nn.Module):
    """TargetDiff-compatible joint protein/ligand hidden encoder."""

    def __init__(self, *, hidden_dim: int = 128, protein_feature_dim: int = 27,
                 ligand_classes: int = 13, num_blocks: int = 1, num_layers: int = 9,
                 heads: int = 16, knn: int = 32, num_rbf: int = 20,
                 r_max: float = 10.0) -> None:
        super().__init__()
        expected = dict(hidden_dim=128, protein_feature_dim=27, ligand_classes=13,
                        num_blocks=1, num_layers=9, heads=16, knn=32, num_rbf=20, r_max=10.0)
        actual = dict(hidden_dim=hidden_dim, protein_feature_dim=protein_feature_dim,
                      ligand_classes=ligand_classes, num_blocks=num_blocks,
                      num_layers=num_layers, heads=heads, knn=knn, num_rbf=num_rbf, r_max=r_max)
        if actual != expected:
            raise ValueError('TargetDiff encoder contract is frozen: expected %r, got %r' % (expected, actual))
        if not TARGETDIFF_ROOT.is_dir():
            raise ImportError('official TargetDiff source is missing: %s' % TARGETDIFF_ROOT)
        if str(TARGETDIFF_ROOT) not in sys.path:
            sys.path.insert(0, str(TARGETDIFF_ROOT))
        try:
            module = importlib.import_module('models.uni_transformer')
            transformer = module.UniTransformerO2TwoUpdateGeneral
        except Exception as exc:
            raise ImportError('cannot import official TargetDiff UniTransformer') from exc
        self.protein_embedding = nn.Linear(protein_feature_dim, 127)
        self.ligand_embedding = nn.Linear(ligand_classes, 127)
        self.node_indicator = nn.Parameter(torch.zeros(2, 1))
        self.encoder = transformer(
            num_blocks=num_blocks, num_layers=num_layers, hidden_dim=hidden_dim,
            n_heads=heads, k=knn, num_r_gaussian=num_rbf, edge_feat_dim=4,
            num_node_types=8, act_fn='relu', norm=True, cutoff_mode='knn',
            ew_net_type='r', num_init_x2h=1, num_init_h2x=0, num_x2h=1,
            num_h2x=1, r_max=r_max, x2h_out_fc=True, sync_twoup=False,
        )
        self.contract = TARGETDIFF_ENCODER_CONTRACT
        self.config = expected

    @staticmethod
    def _validate(protein_pos, protein_feature, batch_protein, ligand_pos,
                  ligand_v, batch_ligand):
        if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3 or protein_pos.dtype != torch.float32:
            raise ValueError('protein_pos must be float32 [Np,3]')
        if ligand_pos.ndim != 2 or ligand_pos.shape[-1] != 3 or ligand_pos.dtype != torch.float32:
            raise ValueError('ligand_pos must be float32 [Nl,3]')
        if protein_feature.shape != (protein_pos.shape[0], 27) or not protein_feature.is_floating_point():
            raise ValueError('protein_feature must be floating [Np,27]')
        if ligand_v.dtype != torch.long or ligand_v.shape != (ligand_pos.shape[0],):
            raise ValueError('ligand_v must be LongTensor [Nl]')
        if ligand_v.numel() and (int(ligand_v.min()) < 0 or int(ligand_v.max()) >= 13):
            raise ValueError('ligand_v must lie in [0,12]')
        for name, value, count in (('batch_protein', batch_protein, protein_pos.shape[0]),
                                   ('batch_ligand', batch_ligand, ligand_pos.shape[0])):
            if value.dtype != torch.long or value.shape != (count,):
                raise ValueError('%s must be LongTensor [%d]' % (name, count))
            if value.numel() and int(value.min()) < 0:
                raise ValueError('%s cannot contain negative ids' % name)
        if protein_pos.device != ligand_pos.device or protein_pos.device != protein_feature.device:
            raise ValueError('encoder tensors must share a device')
        if any(not torch.isfinite(v).all() for v in (protein_pos, ligand_pos, protein_feature)):
            raise ValueError('encoder inputs must be finite')
        num_graphs = int(torch.cat((batch_protein, batch_ligand)).max()) + 1
        if batch_protein.numel() and int(batch_protein.max()) >= num_graphs:
            raise ValueError('invalid protein graph id')
        if batch_ligand.numel() and int(batch_ligand.max()) >= num_graphs:
            raise ValueError('invalid ligand graph id')
        return num_graphs

    def forward(self, protein_pos, protein_feature, batch_protein, ligand_pos,
                ligand_v, batch_ligand) -> Dict[str, torch.Tensor]:
        self._validate(protein_pos, protein_feature, batch_protein, ligand_pos,
                       ligand_v, batch_ligand)
        protein_h = self.protein_embedding(protein_feature)
        ligand_h = self.ligand_embedding(F.one_hot(ligand_v, num_classes=13).float())
        protein_h = torch.cat((protein_h, self.node_indicator[0].expand(protein_h.shape[0], 1)), dim=-1)
        ligand_h = torch.cat((ligand_h, self.node_indicator[1].expand(ligand_h.shape[0], 1)), dim=-1)
        hidden = torch.cat((protein_h, ligand_h), dim=0)
        positions = torch.cat((protein_pos, ligand_pos), dim=0)
        batch = torch.cat((batch_protein, batch_ligand), dim=0)
        mask_ligand = torch.cat((torch.zeros(protein_pos.shape[0], dtype=torch.bool, device=positions.device),
                                 torch.ones(ligand_pos.shape[0], dtype=torch.bool, device=positions.device)))
        before = positions.clone()
        result = self.encoder(hidden, positions, mask_ligand, batch, return_all=False, fix_x=True)
        if not torch.equal(positions, before) or not torch.equal(result['x'], before):
            raise RuntimeError('official encoder changed coordinates despite fix_x=True')
        if not torch.isfinite(result['h']).all():
            raise FloatingPointError('TargetDiff encoder output contains non-finite values')
        split = protein_pos.shape[0]
        return dict(protein_hidden=result['h'][:split], ligand_hidden=result['h'][split:],
                    coordinates=result['x'], edge_contract=torch.tensor([4], device=positions.device))


__all__ = ['TARGETDIFF_ENCODER_CONTRACT', 'TargetDiffEncoderAdapter']
