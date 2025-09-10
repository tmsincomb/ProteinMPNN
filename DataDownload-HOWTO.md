# install original ProteinMPNN
### how ProteinMPNN filters RCSB
```
-   Training Dataset Source

  ProteinMPNN was trained on PDB biounits 
  (biological assemblies) from August 2, 2021,
  curated by Ivan Anishchenko. The dataset is
  16.5 GB and available at https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz.

  Dataset Preparation Pipeline

  1. Structure Selection Criteria

  - Only X-ray crystallography or cryo-EM
  structures
  - Resolution better than 3.5 Å
  - Less than 10,000 residues total length
  - Used biological assemblies (biounits)
  rather than asymmetric units to ensure
  functional relevance

  2. Data Processing

  Each PDB entry was converted to PyTorch .pt
  files containing:
  - Atomic coordinates [L,14,3] for backbone
  and sidechain atoms
  - Amino acid sequences
  - Boolean masks for missing atoms
  - Temperature factors and occupancy
  - Biounit transformation matrices for
  multi-chain assemblies
  - Chain similarity metrics (TM-score,
  sequence identity, RMSD)

  3. Clustering and Train-Test Split

  - Sequences clustered at 30% sequence 
  identity using MMseqs2
  - Clusters assigned to train/validation/test
  sets to prevent data leakage
  - Ensures no similar sequences appear across
  different splits

  4. Noise Augmentation

  - Added Gaussian noise (σ=0.02-0.20 Å) to
  backbone coordinates during training
  - Improves model robustness and prevents
  memorization
  - Different noise levels for different model
  variants (v_48_002, v_48_010, v_48_020)

  5. Graph Representation

  - Proteins converted to k-nearest neighbor
  graphs (k=48 edges)
  - Nodes represent amino acid residues
  - Edges encode spatial relationships and
  distances

  The key insight is using biological 
  assemblies rather than raw PDB structures,
  ensuring the model learns from functionally
  relevant protein complexes as they exist in
  nature.
```

# Usage

```bash
python download_pdb_for_proteinmpnn.py --date-to 2021-08-02 --output-dir rcsb_2021aug02-2025sep08
python format_pdb_download.py --input-dir rcsb_2021aug02-2025sep08 --output-dir pdb_2021aug02-2025sep08 
python merge_pdb_datasets.py --input-dirs pdb_2021aug02-2025sep08 pdb_2021aug02-2025sep08  -output-dir pdb_2025sep08
```