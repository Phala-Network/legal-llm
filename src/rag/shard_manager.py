import json
import hashlib
import os
import sys
from typing import Dict, List, Set, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.path_utils import normalize_case_path

class ShardManager:
    def __init__(self, num_shards: int = 100, assignments_path: str = "shard_assignments.json"):
        self.num_shards = num_shards
        self.assignments_path = assignments_path
        self.case_to_shards: Dict[str, Set[int]] = {} # case_id -> set of shard indices where it must be indexed
        self.primary_shard: Dict[str, int] = {} # case_id -> its primary shard

    def generate_assignments(self, neighborhoods_path: str):
        """
        Generates shard assignments based on citation neighborhoods.
        Each case X has a primary shard.
        Any case Y that is a neighbor of X is also added to X's primary shard.
        """
        if not os.path.exists(neighborhoods_path):
            print(f"Error: {neighborhoods_path} not found.")
            return

        print(f"Loading neighborhoods from {neighborhoods_path}...")
        with open(neighborhoods_path, 'r', encoding='utf-8') as f:
            neighborhoods = json.load(f)

        print("Assigning cases to shards...")
        # 1. Assign each case to a primary shard using hashing for load balance
        for case_id in neighborhoods.keys():
            # Use deterministic hash
            h = int(hashlib.md5(case_id.encode()).hexdigest(), 16)
            primary = h % self.num_shards
            self.primary_shard[case_id] = primary

            if case_id not in self.case_to_shards:
                self.case_to_shards[case_id] = set()
            self.case_to_shards[case_id].add(primary)

        # 2. For each case, ensure all its neighbors are also in its primary shard
        for case_id, neighbors in neighborhoods.items():
            primary = self.primary_shard[case_id]
            for neighbor_id in neighbors:
                if neighbor_id not in self.case_to_shards:
                    self.case_to_shards[neighbor_id] = set()
                self.case_to_shards[neighbor_id].add(primary)

        self.save_assignments()

    def save_assignments(self):
        # Convert set to list for JSON serialization
        serializable = {
            "meta": {
                "num_shards": self.num_shards,
            },
            "primary_shards": self.primary_shard,
            "case_to_shards": {k: list(v) for k, v in self.case_to_shards.items()}
        }
        with open(self.assignments_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
        print(f"Assignments saved to {self.assignments_path}")

    def load_assignments(self) -> bool:
        if not os.path.exists(self.assignments_path):
            return False
        with open(self.assignments_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.num_shards = data["meta"]["num_shards"]
            self.primary_shard = data["primary_shards"]
            self.case_to_shards = {k: set(v) for k, v in data["case_to_shards"].items()}
        return True

    def get_shards_for_case(self, case_id: str) -> List[int]:
        """Returns the list of shard IDs where this case should be indexed."""
        case_id = normalize_case_path(case_id)
        return list(self.case_to_shards.get(case_id, []))

    def get_primary_shard(self, case_id: str) -> Optional[int]:
        """Returns the primary shard for a case, used for searching."""
        case_id = normalize_case_path(case_id)
        return self.primary_shard.get(case_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighborhoods", type=str, default="data/case_neighborhoods.json")
    parser.add_argument("--shards", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/shard_assignments.json")
    args = parser.parse_args()

    sm = ShardManager(num_shards=args.shards, assignments_path=args.output)
    sm.generate_assignments(args.neighborhoods)
