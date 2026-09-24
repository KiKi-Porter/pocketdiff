"""Versioned contracts for the PocketDiff v4.1 residue model."""

NUM_CHI = 4
TARGETDIFF_RESIDUE_NAMES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)
TARGETDIFF_RESIDUE_IDS = {
    name: index for index, name in enumerate(TARGETDIFF_RESIDUE_NAMES)
}
