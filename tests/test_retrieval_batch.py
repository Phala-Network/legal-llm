import os
import argparse
import sys
import json

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""
Note: This test requires a populated test index.
Before running, ensure you have ingested the sample data:
uv run src/rag/ingest.py --search_dir data/sd/6 --db_path chroma_db_verify --index_dir tantivy_index_verify --assignments data/shard_assignments.json
"""

from src.rag.retriever import CaseRetriever

def run_retrieval_batch(db_path="chroma_db", index_dir="tantivy_index", assignments="data/shard_assignments.json"):
    print(f"Initializing CaseRetriever with db_path={db_path}, index_dir={index_dir}...")

    if not os.path.exists(db_path) or not os.path.exists(index_dir):
        print(f"Warning: Test indexes not found at {db_path} or {index_dir}. Skipping batch test.")
        return

    retriever = CaseRetriever(
        db_path=db_path,
        index_dir=index_dir,
        shard_assignments=assignments
    )

    test_queries = [
        {
            "name": "Telegraph Liability",
            "query": "telegraph company liability for failing to send message",
            "expected": "Kirby v. Western Union Telegraph Co."
        },
        {
            "name": "Railroad Negligence (Passenger)",
            "query": "presumption of negligence railroad passenger injury shock",
            "expected": "Saunders v. Chi. & N. W. Ry. Co."
        },
        {
            "name": "Railroad Fire Negligence",
            "query": "railroad section men fire negligence right of way",
            "expected": "Mattoon v. Fremont, E. & M. V. R."
        },
        {
            "name": "Modern Crypto (Negative)",
            "query": "modern crypto currency regulation 2024 blockchain",
            "expected": None
        },
        {
            "name": "Maritime Law (Negative)",
            "query": "maritime salvage rights admiralty law high seas",
            "expected": None
        }
    ]

    print("\nRunning Retrieval Batch Tests...\n")

    all_pass = True
    for i, test in enumerate(test_queries):
        print(f"TEST {i+1}: {test['name']} - '{test['query']}'")
        results = retriever.retrieve(test['query'], k=5)

        if not results:
            if test['expected'] is None:
                print(">>> PASS: No results as expected for negative query.")
            else:
                print(">>> FAIL: No results returned.")
                all_pass = False
            continue

        top_hit = results[0]
        meta = top_hit.get('metadata', {})
        score = top_hit.get('rerank_score', 0.0)
        name = meta.get('name', 'Unknown')

        print(f"    Top Hit: {name} (Score: {score:.4f})")

        if test['expected']:
            # Relaxed matching
            res_names = [(res.get('metadata', {}).get('name') or "").lower() for res in results[:2]]
            expected_lower = test['expected'].lower()

            found = False
            for res_name in res_names:
                if expected_lower in res_name or res_name in expected_lower:
                    found = True
                    break

            if found:
                print(">>> PASS")
            else:
                print(f">>> FAIL: Expected {test['expected']} in top 2. Found: {res_names}")
                all_pass = False
        else:
            if score < 0.1:
                print(">>> PASS: Low relevance score as expected.")
            else:
                print(f">>> WARN: High relevance score ({score:.4f}) for negative query.")

    return all_pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="chroma_db_verify")
    parser.add_argument("--index_dir", type=str, default="tantivy_index_verify")
    parser.add_argument("--assignments", type=str, default="data/shard_assignments.json")
    args = parser.parse_args()

    success = run_retrieval_batch(args.db_path, args.index_dir, args.assignments)
    sys.exit(0 if success else 1)
