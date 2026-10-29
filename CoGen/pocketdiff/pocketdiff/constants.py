"""Frozen v4.8 data vocabulary contracts."""

NUM_CHI = 4
TARGETDIFF_RESIDUE_NAMES = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)
TARGETDIFF_RESIDUE_IDS = {
    name: index for index, name in enumerate(TARGETDIFF_RESIDUE_NAMES)
}

__all__ = [
    "NUM_CHI",
    "TARGETDIFF_RESIDUE_NAMES",
    "TARGETDIFF_RESIDUE_IDS",
]
