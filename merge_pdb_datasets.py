#!/usr/bin/env python3
"""
Merge multiple PDB datasets with the same hierarchy as pdb_2021aug02.
Prioritizes earlier datasets in the input list when conflicts occur.
"""

import argparse
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Set, Dict
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_pdb_datasets(input_dirs: List[str], output_dir: str, verbose: bool = False):
    """
    Merge multiple PDB datasets into a single dataset.
    
    Args:
        input_dirs: List of input dataset directories (in priority order)
        output_dir: Output directory for merged dataset
        verbose: Enable verbose logging
    """
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Convert to Path objects
    input_paths = [Path(d) for d in input_dirs]
    output_path = Path(output_dir)
    
    # Validate input directories
    for path in input_paths:
        if not path.exists():
            raise ValueError(f"Input directory does not exist: {path}")
        if not (path / "list.csv").exists():
            raise ValueError(f"No list.csv found in: {path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating merged dataset in: {output_path}")
    
    # Track processed files and chains
    processed_files = set()  # Track which files have been copied
    processed_chains = set()  # Track which chain IDs have been processed
    chain_to_source = {}  # Map chain ID to source dataset
    
    # Create pdb subdirectory structure
    pdb_dir = output_path / "pdb"
    pdb_dir.mkdir(exist_ok=True)
    
    # Process datasets in order (first has highest priority)
    logger.info(f"Processing {len(input_paths)} datasets in priority order...")
    
    # Step 1: Copy PDB files (.pt files)
    for dataset_idx, dataset_path in enumerate(input_paths):
        logger.info(f"\nProcessing dataset {dataset_idx + 1}/{len(input_paths)}: {dataset_path}")
        
        dataset_pdb_dir = dataset_path / "pdb"
        if not dataset_pdb_dir.exists():
            logger.warning(f"No pdb directory found in {dataset_path}, skipping PDB files")
            continue
        
        # Find all .pt files
        pt_files = list(dataset_pdb_dir.rglob("*.pt"))
        logger.info(f"Found {len(pt_files)} .pt files")
        
        copied_count = 0
        skipped_count = 0
        
        for pt_file in tqdm(pt_files, desc=f"Copying from dataset {dataset_idx + 1}"):
            # Get relative path from pdb directory
            rel_path = pt_file.relative_to(dataset_pdb_dir)
            
            # Create subdirectory if needed (e.g., pdb/ab/)
            output_file = pdb_dir / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists (prioritize earlier datasets)
            if output_file.exists():
                skipped_count += 1
                logger.debug(f"Skipping (already exists): {rel_path}")
                continue
            
            # Copy file
            shutil.copy2(pt_file, output_file)
            processed_files.add(str(rel_path))
            copied_count += 1
            
            # Track chain if it's a chain file (e.g., 1abc_A.pt)
            filename = pt_file.stem
            if '_' in filename:
                chain_id = filename
                processed_chains.add(chain_id)
                chain_to_source[chain_id] = str(dataset_path)
        
        logger.info(f"Copied {copied_count} files, skipped {skipped_count} existing files")
    
    # Step 2: Merge list.csv files
    logger.info("\nMerging list.csv files...")
    
    all_dataframes = []
    for dataset_idx, dataset_path in enumerate(input_paths):
        list_csv = dataset_path / "list.csv"
        if not list_csv.exists():
            continue
        
        df = pd.read_csv(list_csv)
        df['source_dataset'] = str(dataset_path)
        df['dataset_priority'] = dataset_idx
        all_dataframes.append(df)
        logger.info(f"Loaded {len(df)} chains from {dataset_path}")
    
    if not all_dataframes:
        raise ValueError("No list.csv files found in any input dataset")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Total chains before deduplication: {len(combined_df)}")
    
    # Remove duplicates, keeping the first occurrence (highest priority)
    # Sort by dataset_priority first to ensure we keep the right ones
    combined_df = combined_df.sort_values('dataset_priority')
    combined_df = combined_df.drop_duplicates(subset=['CHAINID'], keep='first')
    
    # Remove helper columns
    combined_df = combined_df.drop(columns=['source_dataset', 'dataset_priority'])
    
    # Save merged list.csv
    output_list = output_path / "list.csv"
    combined_df.to_csv(output_list, index=False)
    logger.info(f"Saved merged list.csv with {len(combined_df)} unique chains")
    
    # Step 3: Merge cluster files
    logger.info("\nMerging cluster files...")
    
    for cluster_file in ['valid_clusters.txt', 'test_clusters.txt']:
        all_clusters = set()
        
        for dataset_path in input_paths:
            cluster_path = dataset_path / cluster_file
            if cluster_path.exists():
                with open(cluster_path, 'r') as f:
                    clusters = {int(line.strip()) for line in f if line.strip()}
                    all_clusters.update(clusters)
                    logger.debug(f"Added {len(clusters)} clusters from {dataset_path}/{cluster_file}")
        
        # Save merged cluster file
        if all_clusters:
            output_cluster = output_path / cluster_file
            with open(output_cluster, 'w') as f:
                for cluster in sorted(all_clusters):
                    f.write(f"{cluster}\n")
            logger.info(f"Saved {cluster_file} with {len(all_clusters)} clusters")
    
    # Step 4: Copy README if exists (from first dataset)
    for dataset_path in input_paths:
        readme_path = dataset_path / "README"
        if readme_path.exists():
            shutil.copy2(readme_path, output_path / "README")
            logger.info(f"Copied README from {dataset_path}")
            break
    
    # Step 5: Create merge summary
    summary_path = output_path / "MERGE_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("PDB Dataset Merge Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Input datasets (in priority order):\n")
        for idx, path in enumerate(input_paths, 1):
            f.write(f"  {idx}. {path}\n")
        f.write(f"\nTotal unique chains: {len(combined_df)}\n")
        f.write(f"Total .pt files: {len(processed_files)}\n")
        
        # Chain distribution by source
        f.write(f"\nChains by source dataset:\n")
        for dataset_path in input_paths:
            count = sum(1 for v in chain_to_source.values() if v == str(dataset_path))
            f.write(f"  {dataset_path}: {count} chains\n")
    
    logger.info(f"\nMerge complete! Summary saved to {summary_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("MERGE COMPLETE")
    print("=" * 50)
    print(f"Output directory: {output_path}")
    print(f"Total unique chains: {len(combined_df)}")
    print(f"Total .pt files copied: {len(processed_files)}")
    print("\nYou can now use this merged dataset for training:")
    print(f"  cd training")
    print(f"  python training.py --path_for_training_data ../{output_dir} ...")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple PDB datasets with the same hierarchy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge two datasets
  python merge_pdb_datasets.py -i pdb_2021aug02 pdb_new -o pdb_merged
  
  # Merge with verbose output
  python merge_pdb_datasets.py -i dataset1 dataset2 dataset3 -o merged -v
  
Priority:
  Earlier datasets in the input list have higher priority.
  When files conflict, the version from the earlier dataset is kept.
  
Directory Structure:
  Input datasets should have the structure:
    dataset/
      ├── list.csv
      ├── pdb/
      │   ├── 00/
      │   │   ├── 100d.pt
      │   │   └── ...
      │   └── ...
      ├── valid_clusters.txt
      └── test_clusters.txt
        """
    )
    
    parser.add_argument(
        "-i", "--input-dirs",
        nargs='+',
        required=True,
        help="Input dataset directories in priority order (first = highest priority)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for merged dataset"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    args = parser.parse_args()
    
    # Check if output directory exists
    if Path(args.output_dir).exists() and not args.force:
        response = input(f"Output directory {args.output_dir} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Merge cancelled.")
            return
        # Remove existing directory
        shutil.rmtree(args.output_dir)
    
    try:
        merge_pdb_datasets(
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()