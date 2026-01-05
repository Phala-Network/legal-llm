import os
import json
import argparse
from typing import Dict, List, Set
from collections import Counter, defaultdict
import tqdm
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.path_utils import normalize_case_path


def get_case_id_from_file(rel_path: str, base_dir: str) -> str:
    return normalize_case_path(rel_path)

def analyze_neighborhoods(base_dir: str, search_dir: str, output_file: str = None):
    print(f"Base directory: {base_dir}")
    print(f"Search directory: {search_dir}")

    case_files = []
    case_ids = set()

    # Adjacency list for the citation graph (undirected)
    adj = defaultdict(set)

    # 1. Collect all case files and IDs
    print("Scanning for case files...")
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".json") and "Metadata" not in file:
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                case_id = get_case_id_from_file(rel_path, base_dir)
                case_files.append((case_id, os.path.join(root, file)))
                case_ids.add(case_id)

    print(f"Found {len(case_ids)} unique cases.")

    # 2. Build the citation graph
    print("Building adjacency list...")
    for case_id, file_path in tqdm.tqdm(case_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cites_to = data.get("cites_to", [])
            for cite in cites_to:
                paths = cite.get("case_paths", [])
                for p in paths:
                    cited_id = normalize_case_path(p)
                    # We only care about connections within our searchable set for this experiment
                    if cited_id in case_ids and cited_id != case_id:
                        adj[case_id].add(cited_id)
                        adj[cited_id].add(case_id)
        except Exception:
            continue

    # 3. Analyze neighborhood sizes
    print("\nCalculating neighborhood statistics...")
    # Neighborhood(X) = {X} + direct neighbors
    neighborhood_sizes = []
    neighborhoods = {}

    for case_id in case_ids:
        neighbors = adj[case_id]
        size = len(neighbors) + 1 # Include self
        neighborhood_sizes.append(size)
        if output_file:
            neighborhoods[case_id] = sorted(list(neighbors | {case_id}))

    size_counts = Counter(neighborhood_sizes)
    total_cases = len(case_ids)

    print("\nResult Statistics:")
    print(f"Total Cases processed: {total_cases}")
    print(f"Max Neighborhood Size: {max(neighborhood_sizes) if neighborhood_sizes else 0}")
    print(f"Average Neighborhood Size: {sum(neighborhood_sizes)/total_cases:.2f}" if total_cases > 0 else "N/A")

    print("\nNeighborhood Size Distribution:")
    ranges = [
        (1, 1),
        (2, 5),
        (6, 10),
        (11, 20),
        (21, 50),
        (51, 100),
        (101, 500),
        (501, float('inf'))
    ]

    for start, end in ranges:
        count = sum(c for s, c in size_counts.items() if start <= s <= end)
        label = f"{start}-{end}" if end != float('inf') else f"{start}+"
        if start == end: label = f"{start}"
        percentage = (count / total_cases) * 100 if total_cases > 0 else 0
        print(f"  Size {label:10}: {count:8} cases ({percentage:6.2f}%)")

    # 4. Save results
    if output_file:
        print(f"\nSaving results to {output_file}...")
        # Sort by size to make debugging easier
        sorted_output = {
            k: v for k, v in sorted(neighborhoods.items(), key=lambda x: len(x[1]), reverse=True)
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_output, f, indent=2)
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--output", type=str, help="Path to save neighborhood data (JSON)")
    args = parser.parse_args()

    analyze_neighborhoods(args.base_dir, args.search_dir, args.output)
