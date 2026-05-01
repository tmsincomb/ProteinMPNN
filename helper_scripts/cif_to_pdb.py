#!/usr/bin/env python3
"""Convert mmCIF to PDB for ProteinMPNN inference.

ProteinMPNN's inference pipeline (`protein_mpnn_run.py`) only parses
fixed-column PDB. CIF parsing in this repo is training-only
(`training/parse_cif_noX.py`). When you have a CIF you want to design
sequences for, convert it first with this helper.

Usage:
    python helper_scripts/cif_to_pdb.py input.cif
    python helper_scripts/cif_to_pdb.py input.cif --out_path out.pdb
    python helper_scripts/cif_to_pdb.py input.cif --use_auth_chains

Requires Biopython (`pip install biopython`).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


def convert(cif_path: Path, pdb_path: Path, use_auth_chains: bool) -> dict:
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError:
        print(
            "Biopython is not installed. Install with: pip install biopython",
            file=sys.stderr,
        )
        sys.exit(2)

    warnings.simplefilter("ignore")
    parser = MMCIFParser(QUIET=True, auth_chains=use_auth_chains)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    chains = sorted({c.id for m in structure for c in m})
    n_residues = sum(
        1 for m in structure for c in m for r in c if r.id[0] == " "
    )
    n_atoms = sum(1 for m in structure for c in m for r in c for _ in r)
    n_models = len(list(structure))

    if n_atoms > 99_999:
        print(
            f"WARNING: {n_atoms} atoms exceeds the PDB single-file limit "
            "of 99,999. The output PDB will likely be truncated. "
            "Consider splitting chains or using only the relevant subset.",
            file=sys.stderr,
        )

    io = PDBIO()
    io.set_structure(structure)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(pdb_path))

    return {
        "models": n_models,
        "chains": chains,
        "residues": n_residues,
        "atoms": n_atoms,
        "out_size_mb": pdb_path.stat().st_size / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mmCIF to PDB for ProteinMPNN.",
    )
    parser.add_argument(
        "cif_path",
        type=Path,
        help="Path to the input .cif (or .cif.gz) file.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help=(
            "Path to the output .pdb file. Default: same directory as the "
            "input, with the .cif extension replaced by .pdb."
        ),
    )
    parser.add_argument(
        "--use_auth_chains",
        action="store_true",
        help=(
            "Use mmCIF auth_asym_id (PDB-style chain IDs) instead of "
            "label_asym_id. Default: True in newer Biopython; pass this "
            "flag to be explicit."
        ),
    )
    args = parser.parse_args()

    cif_path = args.cif_path.expanduser().resolve()
    if not cif_path.exists():
        print(f"Input file not found: {cif_path}", file=sys.stderr)
        sys.exit(1)

    if args.out_path is None:
        pdb_path = cif_path.with_suffix(".pdb")
    else:
        pdb_path = args.out_path.expanduser().resolve()

    info = convert(cif_path, pdb_path, args.use_auth_chains)

    print(f"Input:    {cif_path}")
    print(f"Output:   {pdb_path} ({info['out_size_mb']:.1f} MB)")
    print(f"Models:   {info['models']}")
    print(f"Chains:   {len(info['chains'])} -> {info['chains']}")
    print(f"Residues: {info['residues']}")
    print(f"Atoms:    {info['atoms']}")


if __name__ == "__main__":
    main()
