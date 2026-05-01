#!/usr/bin/env python3
"""End-to-end MPS smoke test for ProteinMPNN.

Runs `protein_mpnn_run.py` as a subprocess against the smallest bundled PDB
on the MPS device, then verifies:

1. The device-selection banner names MPS (proves GPU was used).
2. Exit code is 0.
3. A non-empty FASTA file with at least one designed sequence is written.

Skips cleanly with a clear message when MPS is not available, so this file
is safe to run on CI Linux as well.

Run:
    python test_mps_e2e.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent
SMALL_PDB = REPO_ROOT / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"
RUNNER = REPO_ROOT / "protein_mpnn_run.py"

MPS_BANNER = "Using Apple Metal Performance Shaders (MPS)"
TIMEOUT_SECONDS = 180  # First MPS run JIT-compiles shaders; allow headroom.


def _skip(message: str) -> NoReturn:
    print(f"SKIP: {message}")
    sys.exit(0)


def _fail(message: str) -> NoReturn:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    try:
        import torch
    except ImportError:
        _skip("PyTorch is not installed.")

    if not torch.backends.mps.is_available():
        _skip(
            "MPS is not available on this machine. "
            "This test only runs on Apple Silicon with PyTorch >= 1.12."
        )

    if not RUNNER.exists():
        _fail(f"Runner not found: {RUNNER}")
    if not SMALL_PDB.exists():
        _fail(f"Test PDB not found: {SMALL_PDB}")

    out_dir = Path(tempfile.mkdtemp(prefix="mpnn_mps_e2e_"))
    print(f"Output dir: {out_dir}")
    print(f"PDB: {SMALL_PDB}")

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--pdb_path", str(SMALL_PDB),
        "--out_folder", str(out_dir),
        "--num_seq_per_target", "2",
        "--batch_size", "1",
        "--sampling_temp", "0.1",
        "--seed", "37",
    ]

    print("Command:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        _fail(f"Subprocess timed out after {TIMEOUT_SECONDS}s: {exc}")

    print("--- stdout ---")
    print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    if result.returncode != 0:
        shutil.rmtree(out_dir, ignore_errors=True)
        _fail(f"Subprocess exited with code {result.returncode}.")

    if MPS_BANNER not in result.stdout:
        shutil.rmtree(out_dir, ignore_errors=True)
        _fail(
            f"Device banner '{MPS_BANNER}' not found in stdout. "
            "Check that MPS device-selection logic is wired correctly."
        )

    fasta_files = list(out_dir.rglob("*.fa"))
    if not fasta_files:
        shutil.rmtree(out_dir, ignore_errors=True)
        _fail("No FASTA output files produced.")

    fasta = fasta_files[0]
    sequences = [
        line.strip()
        for line in fasta.read_text().splitlines()
        if line and not line.startswith(">")
    ]
    if not sequences:
        shutil.rmtree(out_dir, ignore_errors=True)
        _fail(f"FASTA file {fasta} contains no sequences.")

    print(f"OK: {fasta} ({len(sequences)} sequence lines)")
    print("PASS: MPS end-to-end smoke test succeeded.")
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
