#!/usr/bin/env python3
"""
Download PDB structures from RCSB that meet ProteinMPNN training criteria.
Efficiently queries and downloads structures using RCSB Search API v2.
"""

import os
import json
import time
import requests
import argparse
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RCSBDownloader:
    """Download PDB structures meeting ProteinMPNN criteria from RCSB."""
    
    def __init__(self, output_dir: str = "pdb_download", max_workers: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.search_api = "https://search.rcsb.org/rcsbsearch/v2/query"
        self.download_api = "https://files.rcsb.org/download"
        
    def build_search_query(self, 
                          max_resolution: float = 3.5,
                          max_length: int = 10000,
                          methods: List[str] = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
                          date_from: str = None,
                          date_to: str = None) -> Dict:
        """
        Build RCSB search query matching ProteinMPNN criteria.
        
        Args:
            max_resolution: Maximum resolution in Angstroms (default: 3.5)
            max_length: Maximum total polymer length (default: 10000)
            methods: Experimental methods to include
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        """
        
        # Base query for experimental method and resolution
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": []
            },
            "request_options": {
                "return_all_hits": True
            },
            "return_type": "entry"
        }
        
        # Add experimental method filter
        if methods:
            method_node = {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": method
                        }
                    } for method in methods
                ]
            }
            query["query"]["nodes"].append(method_node)
        
        # Add resolution filter
        if max_resolution:
            resolution_node = {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": max_resolution
                }
            }
            query["query"]["nodes"].append(resolution_node)
        
        # Add polymer length filter
        if max_length:
            length_node = {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                    "operator": "less_or_equal",
                    "value": max_length
                }
            }
            query["query"]["nodes"].append(length_node)
        
        # Add date range filter if specified
        if date_from or date_to:
            date_nodes = []
            if date_from:
                date_nodes.append({
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": date_from
                    }
                })
            if date_to:
                date_nodes.append({
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less_or_equal",
                        "value": date_to
                    }
                })
            
            if date_nodes:
                date_group = {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": date_nodes
                }
                query["query"]["nodes"].append(date_group)
        
        return query
    
    def search_structures(self, query: Dict) -> List[str]:
        """Execute search query and return PDB IDs."""
        
        logger.info("Searching RCSB with ProteinMPNN criteria...")
        
        try:
            response = requests.post(
                self.search_api,
                json=query,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            pdb_ids = [item["identifier"] for item in data.get("result_set", [])]
            
            logger.info(f"Found {len(pdb_ids)} structures matching criteria")
            return pdb_ids
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def download_pdb(self, pdb_id: str, format: str = "pdb") -> bool:
        """
        Download a single PDB file.
        
        Args:
            pdb_id: 4-letter PDB ID
            format: File format (pdb, cif, or pdb1 for biological assembly)
        """
        
        # Create subdirectory using PDB ID sharding (2nd and 3rd characters)
        subdir = self.output_dir / pdb_id[1:3].lower()
        subdir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        if format == "pdb1":
            # Biological assembly
            filename = f"{pdb_id.lower()}.pdb1.gz"
            url = f"{self.download_api}/{pdb_id.upper()}.pdb1.gz"
        else:
            filename = f"{pdb_id.lower()}.{format}.gz"
            url = f"{self.download_api}/{pdb_id.upper()}.{format}.gz"
        
        gz_path = subdir / filename
        final_path = subdir / filename.replace('.gz', '')
        
        # Skip if already downloaded
        if final_path.exists():
            return True
        
        try:
            # Download compressed file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save compressed file
            with open(gz_path, 'wb') as f:
                f.write(response.content)
            
            # Decompress
            with gzip.open(gz_path, 'rb') as f_in:
                with open(final_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove compressed file
            gz_path.unlink()
            
            return True
            
        except Exception as e:
            logger.debug(f"Failed to download {pdb_id}: {e}")
            return False
    
    def download_batch(self, pdb_ids: List[str], format: str = "pdb", 
                      use_biounit: bool = True) -> Dict[str, bool]:
        """
        Download multiple PDB files in parallel.
        
        Args:
            pdb_ids: List of PDB IDs
            format: File format
            use_biounit: Download biological assemblies (pdb1) instead of asymmetric units
        """
        
        results = {}
        download_format = "pdb1" if use_biounit else format
        
        logger.info(f"Downloading {len(pdb_ids)} structures...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_pdb = {
                executor.submit(self.download_pdb, pdb_id, download_format): pdb_id
                for pdb_id in pdb_ids
            }
            
            # Process completed downloads
            with tqdm(total=len(pdb_ids), desc="Downloading") as pbar:
                for future in as_completed(future_to_pdb):
                    pdb_id = future_to_pdb[future]
                    try:
                        success = future.result()
                        results[pdb_id] = success
                    except Exception as e:
                        logger.error(f"Error downloading {pdb_id}: {e}")
                        results[pdb_id] = False
                    pbar.update(1)
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Successfully downloaded {successful}/{len(pdb_ids)} structures")
        
        return results
    
    def download_recent_updates(self, days: int = 30, **kwargs) -> Dict[str, bool]:
        """
        Download structures released in the last N days.
        
        Args:
            days: Number of days to look back
            **kwargs: Additional search criteria
        """
        
        # Calculate date range
        date_to = datetime.now().strftime("%Y-%m-%d")
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Build and execute query
        query = self.build_search_query(date_from=date_from, date_to=date_to, **kwargs)
        pdb_ids = self.search_structures(query)
        
        if not pdb_ids:
            logger.info("No new structures found")
            return {}
        
        # Download structures
        return self.download_batch(pdb_ids)
    
    def save_metadata(self, pdb_ids: List[str], filename: str = "download_metadata.json"):
        """Save metadata about downloaded structures."""
        
        metadata = {
            "download_date": datetime.now().isoformat(),
            "total_structures": len(pdb_ids),
            "pdb_ids": sorted(pdb_ids),
            "criteria": {
                "max_resolution": 3.5,
                "max_length": 10000,
                "methods": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
            }
        }
        
        metadata_path = self.output_dir / filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PDB structures meeting ProteinMPNN criteria"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="pdb_download",
        help="Output directory for downloaded files"
    )
    parser.add_argument(
        "--max-resolution", "-r",
        type=float,
        default=3.5,
        help="Maximum resolution in Angstroms (default: 3.5)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=10000,
        help="Maximum polymer length (default: 10000)"
    )
    parser.add_argument(
        "--date-from",
        help="Start date for structures (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date-to",
        help="End date for structures (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        help="Download structures from last N days (overrides date range)"
    )
    parser.add_argument(
        "--use-biounit",
        action="store_true",
        default=True,
        help="Download biological assemblies (default: True)"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=10,
        help="Maximum parallel downloads (default: 10)"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of structures to download"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = RCSBDownloader(
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    if args.recent_days:
        # Download recent structures
        from datetime import timedelta
        results = downloader.download_recent_updates(
            days=args.recent_days,
            max_resolution=args.max_resolution,
            max_length=args.max_length
        )
    else:
        # Build search query
        query = downloader.build_search_query(
            max_resolution=args.max_resolution,
            max_length=args.max_length,
            date_from=args.date_from,
            date_to=args.date_to
        )
        
        # Search for structures
        pdb_ids = downloader.search_structures(query)
        
        if args.limit:
            pdb_ids = pdb_ids[:args.limit]
            logger.info(f"Limiting to {args.limit} structures")
        
        if pdb_ids:
            # Download structures
            results = downloader.download_batch(
                pdb_ids,
                use_biounit=args.use_biounit
            )
            
            # Save metadata
            downloader.save_metadata(pdb_ids)
        else:
            logger.info("No structures found matching criteria")
            results = {}
    
    # Print summary
    if results:
        successful = sum(1 for v in results.values() if v)
        print(f"\nDownload complete: {successful}/{len(results)} structures")
        print(f"Files saved to: {downloader.output_dir}")


if __name__ == "__main__":
    # Example usage matching ProteinMPNN training dataset criteria:
    #
    # 1. Download recent structures (last 30 days) with ProteinMPNN criteria:
    #    python download_pdb_for_proteinmpnn.py --recent-days 30
    #
    # 2. Download all structures up to Aug 2, 2021 (original training cutoff):
    #    python download_pdb_for_proteinmpnn.py --date-to 2021-08-02
    #
    # 3. Download with custom criteria:
    #    python download_pdb_for_proteinmpnn.py \
    #        --max-resolution 3.5 \     # Maximum resolution in Angstroms
    #        --max-length 10000 \        # Maximum polymer length
    #        --use-biounit \             # Download biological assemblies (default)
    #        --output-dir pdb_biounits   # YOUR FOLDER: Will be created
    #
    # 4. Download limited set for testing:
    #    python download_pdb_for_proteinmpnn.py --limit 100 --output-dir test_pdbs
    #
    # Note: All examples use default ProteinMPNN criteria:
    #       - Only X-ray crystallography or cryo-EM structures
    #       - Resolution better than 3.5 Å
    #       - Less than 10,000 residues total
    #       - Biological assemblies (biounits) rather than asymmetric units
    
    main()