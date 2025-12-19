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


class CaseRetriever:
    def __init__(self, db_path="chroma_db", collection_name="law_cases"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        self.client = OpenAI()
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large"
        )

        # Reranker
        print("Loading Reranker (this may take a moment first time)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

        # Sparse Index (BM25)
        # Note: In a production system, BM25 index should be persisted separately.
        # For this setup, we'll try to load it from a pickle or rebuild it from Chroma (slow on large data).
        self.bm25_path = "bm25_index.pkl"
        self.bm25 = None
        self.doc_ids = []  # Map index to ID
        self.docs_text = []  # Map index to Text
        self._init_bm25()

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

    def search_vector(self, query, k=25):
        embedding = (
            self.client.embeddings.create(
                input=[query], model=self.embedding_model_name
            )
            .data[0]
            .embedding
        )
        img_results = self.collection.query(query_embeddings=[embedding], n_results=k)

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

    def retrieve(self, query, k=5):
        # 1. Hybrid Retrieval (Vector + Keyword)
        # Get more candidates for reranking
        vector_k = 30
        keyword_k = 30

        vec_results = self.search_vector(query, k=vector_k)
        kw_results = self.search_keyword(query, k=keyword_k)

        # Merge (RRF or simple union)
        # Use Simple ID deduplication map
        combined = {r["id"]: r for r in vec_results}
        for r in kw_results:
            if r["id"] not in combined:
                combined[r["id"]] = r
                # If metadata missing (BM25 path), try to fetch (skipped for speed here, assume vectors catch most)
                # Or simplistic: BM25 hits only valid if in vector DB too? No, that defeats purpose.
                # For this demo, let's assume we rely mostly on Vector metadata or cross-referenced.
                # Actually, failing to return metadata from BM25 hits is bad.
                # Let's just trust Vector for now?
                # Better: In _init_bm25, we should have stored metadatas or we do a bulk get from Chroma for missing IDs.

        # To fix metadata issue for BM25-only hits:
        bm25_only_ids = [rid for rid in combined if not combined[rid].get("metadata")]
        if bm25_only_ids:
            try:
                metas = self.collection.get(
                    ids=bm25_only_ids, include=["metadatas", "documents"]
                )
                for i, mid in enumerate(metas["ids"]):
                    if mid in combined:
                        combined[mid]["metadata"] = metas["metadatas"][i]
                        combined[mid]["text"] = metas["documents"][
                            i
                        ]  # Ensure text is there
            except:
                pass

        candidates = list(combined.values())

        if not candidates:
            return []

        # 2. Re-ranking
        # Pairs of (query, doc_text)
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:k]


if __name__ == "__main__":
    # Auto-detect collection
    temp_client = chromadb.PersistentClient(path="chroma_db")
    cols = [
        c.name for c in temp_client.list_collections() if c.name.startswith("law_cases")
    ]

    if not cols:
        print("No 'law_cases' collections found.")
    else:
        target_col = cols[0]
        print(f"Connecting to collection: {target_col}")

        retriever = CaseRetriever(collection_name=target_col)
        print("\nTest Retrieval:")
        results = retriever.retrieve("probation conditions for minors")
        for r in results:
            print(
                f"[{r['rerank_score']:.3f}] {r['metadata'].get('name', 'Unknown')}: {r['text'][:100]}..."
            )
