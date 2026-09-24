"""Versioned contracts for the PocketDiff v4 residue model."""

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
