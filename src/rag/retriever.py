import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi
import pickle
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

import sys
# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.shard_manager import ShardManager
from src.utils.path_utils import normalize_case_path

class CaseRetriever:
    def __init__(self, db_path="chroma_db", default_collection="global_collection", shard_assignments="data/shard_assignments.json"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.default_collection_name = default_collection
        self.client = OpenAI()
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large"
        )

        # Initialize ShardManager
        self.shard_manager = ShardManager(assignments_path=shard_assignments)
        self.sharding_enabled = self.shard_manager.load_assignments()
        if not self.sharding_enabled:
            print(f"Warning: {shard_assignments} not found. Retrieval will use default collection.")

        # Cache for collections
        self.collections = {}

        # Reranker
        print("Loading Reranker (this may take a moment first time)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        # Sparse Index (BM25)
        # Note: In a sharded system, BM25 should ideally be sharded too.
        # For this version, we'll maintain the existing logic but aim for shard-specific BM25 in future.
        self.bm25_path = "bm25_index.pkl"
        self.bm25 = None
        self.doc_ids = []
        self.docs_text = []
        # self._init_bm25() # Disabling auto-init for now as it needs collection context

    def get_collection(self, name):
        if name not in self.collections:
            try:
                self.collections[name] = self.chroma_client.get_collection(name=name)
            except:
                # If it doesn't exist, we might be searching an empty shard or global
                if name == self.default_collection_name:
                     self.collections[name] = self.chroma_client.create_collection(name=name)
                else:
                     return None
        return self.collections[name]

    def _init_bm25(self):
        if os.path.exists(self.bm25_path):
            print("Loading BM25 index...")
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["model"]
                self.doc_ids = data["ids"]
                self.docs_text = data["texts"]
        else:
            print("Building BM25 index from Vector DB (One-time setup)...")
            # Fetch all documents from Chroma
            # WARNING: This scales poorly. limit to 10k for now or implement scrolling.
            results = self.collection.get()
            texts = results["documents"]
            ids = results["ids"]

            if not texts:
                print("Vector DB empty. BM25 not initialized.")
                return

            tokenized_corpus = [
                doc.split() for doc in texts
            ]  # Simple whitespace tokenizer
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.doc_ids = ids
            self.docs_text = texts

            # Save
            with open(self.bm25_path, "wb") as f:
                pickle.dump({"model": self.bm25, "ids": ids, "texts": texts}, f)
            print("BM25 index built and saved.")

    def search_vector(self, query, k=25, shard_name=None):
        target_col = self.get_collection(shard_name or self.default_collection_name)
        if not target_col:
            return []

        embedding = (
            self.client.embeddings.create(
                input=[query], model=self.embedding_model_name
            )
            .data[0]
            .embedding
        )
        img_results = target_col.query(query_embeddings=[embedding], n_results=k)

        # Format
        hits = []
        if img_results["documents"]:
            for i in range(len(img_results["documents"][0])):
                hits.append(
                    {
                        "id": img_results["ids"][0][i],
                        "text": img_results["documents"][0][i],
                        "metadata": img_results["metadatas"][0][i],
                        "score": 0.0,  # Placeholder
                    }
                )
        return hits

    def search_keyword(self, query, k=25):
        if not self.bm25:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]

        hits = []
        for idx in top_n:
            hits.append(
                {
                    "id": self.doc_ids[idx],
                    "text": self.docs_text[idx],
                    "metadata": {},  # We'd need to store metadata in pickle too to be perfect, or fetch from Chroma
                    "score": scores[idx],
                }
            )
        return hits

    def retrieve(self, query, k=5, focus_case_id=None):
        """
        Retrieves relevant cases. If focus_case_id is provided,
        constrains search to the relevant citation neighborhood shard.
        """
        # Determine target shard
        target_shard = None
        if focus_case_id and self.sharding_enabled:
            focus_case_id = normalize_case_path(focus_case_id)
            shard_id = self.shard_manager.get_primary_shard(focus_case_id)
            if shard_id is not None:
                target_shard = f"shard_{shard_id:03d}"
                print(f"Focusing search on shard: {target_shard} (Case: {focus_case_id})")

        # 1. Hybrid Retrieval (Vector + Keyword)
        vector_k = 30
        keyword_k = 30

        vec_results = self.search_vector(query, k=vector_k, shard_name=target_shard)
        # Note: BM25 keyword search is still simplified for now.
        # Ideally would also be shard-constrained.
        kw_results = self.search_keyword(query, k=keyword_k)

        # Merge
        combined = {r["id"]: r for r in vec_results}
        for r in kw_results:
            if r["id"] not in combined:
                combined[r["id"]] = r

        # Ensure metadata is populated
        missing_metadata_ids = [rid for rid in combined if not combined[rid].get("metadata")]
        if missing_metadata_ids:
            col = self.get_collection(target_shard or self.default_collection_name)
            if col:
                try:
                    metas = col.get(ids=missing_metadata_ids, include=["metadatas", "documents"])
                    for i, mid in enumerate(metas["ids"]):
                         if mid in combined:
                             combined[mid]["metadata"] = metas["metadatas"][i]
                             combined[mid]["text"] = metas["documents"][i]
                except:
                    pass

        candidates = list(combined.values())
        if not candidates:
            return []

        # 2. Re-ranking
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Shard-Aware Retrieval.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--focus", type=str, help="Case ID to focus search (Neighborhood search)")
    parser.add_argument("--db_path", type=str, default="chroma_db")
    parser.add_argument("--assignments", type=str, default="data/shard_assignments.json")
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    retriever = CaseRetriever(
        db_path=args.db_path,
        shard_assignments=args.assignments
    )

    results = retriever.retrieve(args.query, k=args.k, focus_case_id=args.focus)

    print("\n" + "="*50)
    print(f"Results for: '{args.query}'")
    if args.focus:
        print(f"Focus Case: {args.focus}")
    print("="*50 + "\n")

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
