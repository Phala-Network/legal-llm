import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
import numpy as np
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

import sys

# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.shard_manager import ShardManager
from src.retrieval.router import ShardRouter
from src.utils.path_utils import normalize_case_path


class CaseRetriever:
    def __init__(
        self,
        db_path="chroma_db",
        index_dir="tantivy_index",
        shard_assignments="data/shard_assignments.json",
    ):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.client = OpenAI()
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large"
        )

        # Initialize ShardManager
        self.shard_manager = ShardManager(assignments_path=shard_assignments)
        if not self.shard_manager.load_assignments():
            raise ValueError(
                f"Shard assignments not found at {shard_assignments}. Run ingestion first."
            )

        # Initialize Global Router (Tantivy)
        self.router = ShardRouter(index_dir)

        # Cache for collections
        self.collections = {}

        # Reranker
        print("Loading Reranker (this may take a moment first time)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def get_collection(self, name):
        if name not in self.collections:
            try:
                self.collections[name] = self.chroma_client.get_collection(name=name)
            except:
                return None
        return self.collections[name]

    def search_vector(
        self, query: str, k: int = 25, shard_name: str = None
    ) -> List[Dict[str, Any]]:
        target_col = self.get_collection(shard_name)
        if not target_col:
            return []

        embedding = (
            self.client.embeddings.create(
                input=[query], model=self.embedding_model_name
            )
            .data[0]
            .embedding
        )
        results = target_col.query(query_embeddings=[embedding], n_results=k)

        hits = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                hits.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 0.0,
                    }
                )
        return hits

    def retrieve(
        self, query: str, k: int = 5, focus_case_id: str = None, router_top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant cases.
        If focus_case_id is provided, constrains search to that case's primary shard.
        Otherwise, uses the Global Router to find candidate shards.
        """
        target_shards = set()

        if focus_case_id:
            focus_case_id = normalize_case_path(focus_case_id)
            shard_id = self.shard_manager.get_primary_shard(focus_case_id)
            if shard_id is not None:
                target_shards.add(f"shard_{shard_id:03d}")
                print(
                    f"Focusing search on shard: shard_{shard_id:03d} (Case: {focus_case_id})"
                )
        else:
            # Stage 1: Global Routing
            print(f"Routing query: '{query}'")
            candidates = self.router.route(query, top_k=router_top_k)
            for c in candidates:
                slug = c["slug"]
                shard_id = self.shard_manager.get_primary_shard(slug)
                if shard_id is not None:
                    target_shards.add(f"shard_{shard_id:03d}")

            print(f"Routed to shards: {target_shards}")

        if not target_shards:
            print("No target shards identified.")
            return []

        # Stage 2: Vector Search in identified shards
        all_candidates = []
        seen_doc_ids = set()

        for shard_name in target_shards:
            shard_hits = self.search_vector(query, k=20, shard_name=shard_name)
            for hit in shard_hits:
                if hit["id"] not in seen_doc_ids:
                    all_candidates.append(hit)
                    seen_doc_ids.add(hit["id"])

        if not all_candidates:
            return []

        # Stage 3: Re-ranking
        print(f"Re-ranking {len(all_candidates)} candidates...")
        pairs = [[query, doc["text"]] for doc in all_candidates]
        scores = self.reranker.predict(pairs)

        for i, doc in enumerate(all_candidates):
            doc["rerank_score"] = float(scores[i])

        ranked = sorted(all_candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Shard-Aware Retrieval.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument(
        "--focus", type=str, help="Case ID to focus search (Neighborhood search)"
    )
    parser.add_argument("--db_path", type=str, default="chroma_db")
    parser.add_argument("--index_dir", type=str, default="tantivy_index")
    parser.add_argument(
        "--assignments", type=str, default="data/shard_assignments.json"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--router_top_k", type=int, default=10, help="Number of shards to route to"
    )

    args = parser.parse_args()

    retriever = CaseRetriever(
        db_path=args.db_path,
        index_dir=args.index_dir,
        shard_assignments=args.assignments,
    )

    results = retriever.retrieve(
        args.query, k=args.k, focus_case_id=args.focus, router_top_k=args.router_top_k
    )

    print("\n" + "=" * 50)
    print(f"Results for: '{args.query}'")
    if args.focus:
        print(f"Focus Case: {args.focus}")
    print("=" * 50 + "\n")

    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            print(f"{i}. [{r['rerank_score']:.3f}] {meta.get('name', 'Unknown')}")
            print(f"   Case ID: {meta.get('case_id')} | Path: {meta.get('path_id')}")
            print(f"   Citation: {meta.get('citation')}")
            print(f"   Snippet: {r['text'][:200]}...")
            print("-" * 30)
