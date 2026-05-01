# ProteinMPNN on Apple Silicon (MPS) — Operational Notes

This document covers running **this repository** (ProteinMPNN) on Apple Silicon
GPUs via PyTorch's MPS backend. For the related but separate analysis of
porting MMseqs2 to MPS, see [MMseqs2_MPS_Feasibility_Analysis.md](MMseqs2_MPS_Feasibility_Analysis.md).

## Status

- **Inference (`protein_mpnn_run.py`)** — works end-to-end on M-series Macs. Verified on M4 Max with PyTorch 2.2.2 against `inputs/PDB_monomers/pdbs/6MRR.pdb`: 2 sequences of length 68 generated in 2.6 s on the GPU.
- **Device priority** — `cuda → mps → cpu`. No flag needed; device is auto-selected at [protein_mpnn_run.py:68-77](../../protein_mpnn_run.py).
- **Training (`training/training.py`)** — same auto-selection, but not exercised by the smoke test in this repo. Treat as untested on MPS.

## How to run

Use the wrapper at the repo root:

```bash
./run_mpnn_mps.sh \
    --pdb_path inputs/PDB_monomers/pdbs/6MRR.pdb \
    --out_folder outputs/test \
    --num_seq_per_target 8
```

The wrapper sets the env vars described below before invoking
`protein_mpnn_run.py`. You can also export them yourself and call
`python protein_mpnn_run.py` directly.

## Required environment variables

| Variable | Why |
|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | If a PyTorch op is not implemented on MPS, fall back to CPU silently instead of raising `NotImplementedError` mid-design. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` | The default high-watermark caps usable VRAM at ~75% of unified memory (~96 GB on a 128 GB M4 Max). `0.0` removes the cap. Use `0.9` if you want a small safety margin. |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0` | Companion to the above; lets the allocator use the full pool before reclamation kicks in. |

## Allocator hygiene

`protein_mpnn_run.py` calls `torch.mps.empty_cache()` at the top of the outer
protein loop ([protein_mpnn_run.py:238-240](../../protein_mpnn_run.py)). This
prevents the MPS allocator from accumulating tensors across batches, which
would otherwise cause progressive slowdown and eventual OOM during long runs.
The call is guarded so the CUDA and CPU paths are unaffected.

## MPS-specific code fixes applied to this repo

1. **Allocator reset between proteins** — see above.
2. **`bias_AAs_np` dtype** — was `np.zeros(len(alphabet))` (float64). MPS does
   not support float64; the tensor cast inside `ProteinMPNN.sample`
   ([protein_mpnn_utils.py:1135](../../protein_mpnn_utils.py)) failed with
   *"Cannot convert a MPS Tensor to float64 dtype"*. Fixed at the source by
   allocating as `np.float32`, matching the existing convention for
   `omit_AAs_np`.

If you run into other float64 surprises while exercising less-common code
paths (PSSM bias, tied positions, scoring-only modes, custom forks), the
likely culprit is another numpy array allocated without an explicit dtype.
Search for `np.zeros(`, `np.ones(`, `np.array(` calls that flow into
`torch.tensor(..., device=device)` and add `dtype=np.float32` at the source.

## Known gotchas

- **First-run shader compilation.** The first MPS run is slow because Metal
  JIT-compiles shaders. The second run is representative — discard the first
  timing if you benchmark.
- **Float64 unsupported.** Already covered above. Worth re-stating: any new
  numpy array on the inference path must be float32 if it will be moved to
  MPS.
- **`torch.mps.empty_cache()` is op-specific.** It only frees the MPS
  allocator's cache, not host memory. It does not block. Calling it once per
  protein is cheap and safe.
- **Docker on macOS cannot reach Metal.** Containers run inside a Linux VM
  that has no Metal API. Run ProteinMPNN natively (conda/venv) on macOS, not
  inside Docker, if you want GPU acceleration.

## Verification

```bash
# Cheap unit check: PyTorch sees MPS, can place a tensor
python test_mps.py

# End-to-end: runs protein_mpnn_run.py on a real PDB on MPS
python test_mps_e2e.py

# Performance: CPU vs MPS across batch sizes
python benchmark_mps_cpu.py
```

## Memory and batch-size guidance (M4 Max, 128 GB)

These numbers are conservative starting points; profile your own workloads.

| Workload | Batch size | Notes |
|---|---|---|
| Single small monomer (~70 aa) | 1–8 | Sub-second per batch after warmup |
| Typical complex (~500 aa, 2-3 chains) | 4–8 | Good throughput tradeoff |
| Large complex (>1k aa) | 1–4 | Watch peak memory; raise gradually |
| Very large (multi-thousand aa) | 1 | Use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and monitor with `powermetrics` |

Use `--max_length` to bound sequence length when iterating across a dataset.

## See also

- [protein_mpnn_run.py](../../protein_mpnn_run.py) — main inference script (device selection, allocator reset)
- [protein_mpnn_utils.py](../../protein_mpnn_utils.py) — model definition (all standard PyTorch ops)
- [run_mpnn_mps.sh](../../run_mpnn_mps.sh) — Apple Silicon launcher with required env vars
- [test_mps.py](../../test_mps.py) — MPS availability check
- [test_mps_e2e.py](../../test_mps_e2e.py) — end-to-end smoke test on a real PDB
- [benchmark_mps_cpu.py](../../benchmark_mps_cpu.py) — CPU-vs-MPS micro-benchmark
- [MMseqs2_MPS_Feasibility_Analysis.md](MMseqs2_MPS_Feasibility_Analysis.md) — separate, unrelated analysis for MMseqs2
