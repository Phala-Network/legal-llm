import os
import json
import argparse
from typing import Dict, List, Set
from collections import Counter
import tqdm
import sys
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.path_utils import normalize_case_path

class UnionFind:
    def __init__(self, elements: List[str]):
        self.parent = {el: el for el in elements}
        self.size = {el: 1 for el in elements}
        self.num_sets = len(elements)

    def find(self, i: str) -> str:
        if i not in self.parent:
            # Handle cases that are cited but don't exist in our files
            # For this experiment, we only care about connections between existing cases
            return None

        root = i
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression
        while self.parent[i] != root:
            next_node = self.parent[i]
            self.parent[i] = root
            i = next_node
        return root

    def union(self, i: str, j: str):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i and root_j and root_i != root_j:
            # Union by size
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i

            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.num_sets -= 1


def get_case_id_from_file(rel_path: str) -> str:
    # Example: us/1/json/0001-01.json -> us/1/0001-01
    return normalize_case_path(rel_path)

def analyze_groups(base_dir: str, search_dir: str, output_file: str = None):
    print(f"Base directory (for IDs): {base_dir}")
    print(f"Search directory (for files): {search_dir}")
    case_files = []
    case_ids = set()

    # 1. Collect all case files and IDs
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".json") and "Metadata" not in file:
                # Use base_dir to get the ID relative to project data root
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                case_id = get_case_id_from_file(rel_path)
                case_files.append((case_id, os.path.join(root, file)))
                case_ids.add(case_id)

    print(f"Found {len(case_ids)} unique cases to process.")

    uf = UnionFind(list(case_ids))

    # 2. Process citations
    print("Processing citations...")
    for case_id, file_path in tqdm.tqdm(case_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cites_to = data.get("cites_to", [])
            for cite in cites_to:
                paths = cite.get("case_paths", [])
                for p in paths:
                    cited_id = normalize_case_path(p)
                    if cited_id in case_ids:
                        uf.union(case_id, cited_id)
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            continue

    # 3. Analyze results
    print("\nResult Statistics:")
    print(f"Total Cases: {len(case_ids)}")
    print(f"Total Groups (Disjoint Sets): {uf.num_sets}")

    # Group size distribution
    roots = {}
    groups = {} # Map root -> list of member IDs
    for case_id in case_ids:
        root = uf.find(case_id)
        if root not in roots:
            roots[root] = 0
            groups[root] = []
        roots[root] += 1
        groups[root].append(case_id)

    group_sizes = list(roots.values())
    size_counts = Counter(group_sizes)

    max_size = max(group_sizes) if group_sizes else 0
    print(f"Largest Group Size: {max_size}")

    # Show top 10 groups
    print("\nTop 10 Largest Groups:")
    sorted_groups = sorted(roots.items(), key=lambda x: x[1], reverse=True)
    for i, (root, size) in enumerate(sorted_groups[:10], 1):
        print(f"  {i}. Root: {root:30} Size: {size}")

    print("\nGroup Size Distribution:")
    ranges = [
        (1, 1),
        (2, 10),
        (11, 100),
        (101, 1000),
        (1001, 10000),
        (10001, float('inf'))
    ]

    for start, end in ranges:
        count = sum(c for s, c in size_counts.items() if start <= s <= end)
        label = f"{start}-{end}" if end != float('inf') else f"{start}+"
        if start == end: label = f"{start}"
        print(f"  Size {label:10}: {count} groups")

    if output_file:
        print(f"\nSaving grouping results to {output_file}...")
        # Sort groups by size for easier debugging
        sorted_output = {
            root: members for root, members in sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_output, f, indent=2)
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Root data directory for ID normalization (e.g., 'data')")
    parser.add_argument("--search_dir", type=str, required=True, help="Directory to scan for case files (e.g., 'data/us')")
    parser.add_argument("--output", type=str, help="Path to save grouping results (JSON format)")
    args = parser.parse_args()

    analyze_groups(args.base_dir, args.search_dir, args.output)
