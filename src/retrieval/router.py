import tantivy
import os

class ShardRouter:
    def __init__(self, index_dir):
        if not os.path.exists(index_dir):
            raise ValueError(f"Index directory does not exist: {index_dir}")

        # Define Schema (Must match the one used during build)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_unsigned_field("case_id", stored=True)
        schema_builder.add_text_field("slug", stored=True)
        schema = schema_builder.build()

        self.index = tantivy.Index(schema, path=index_dir)
        self.searcher = self.index.searcher()

    def route(self, query_text, top_k=10):
        """
        Routes the query to the most relevant cases.
        Prioritizes cases containing ALL query terms (Tier 1),
        then falls back to standard BM25 scoring (Tier 2).
        """
        # Simple tokenization for query construction
        # Remove special characters that might confuse the parser if we inject them
        safe_query = query_text.replace("+", "").replace("-", "")
        terms = safe_query.split()

        if not terms:
            return []

        candidates = []
        seen_ids = set()

        # Tier 1: Strict Intersection (AND)
        if len(terms) > 1:
            and_query_str = " ".join([f"+{t}" for t in terms])
            try:
                q_and = self.index.parse_query(and_query_str, ["body"])
                # Get more than needed to ensure we fill top_k if high quality exists
                res_and = self.searcher.search(q_and, top_k).hits

                for score, doc_address in res_and:
                    doc = self.searcher.doc(doc_address)
                    c_id = doc['case_id'][0]
                    slug = doc['slug'][0]

                    if c_id not in seen_ids:
                        # Boosting score arbitrarily to indicate Tier 1?
                        # Or just relying on list order. List order is safer.
                        # We keep the original BM25 score but they are first in list.
                        candidates.append({
                            "case_id": c_id,
                            "slug": slug,
                            "score": score,
                            "tier": "strict_and" # Metadata for debugging
                        })
                        seen_ids.add(c_id)
            except Exception as e:
                # If query parsing fails (e.g. stop words only?), ignore
                pass

        # Tier 2: Standard Union (OR) - Fill remaining spots
        if len(candidates) < top_k:
            remaining_k = top_k # We ask for full top_k again to get the best of the rest
            try:
                q_or = self.index.parse_query(safe_query, ["body"])
                res_or = self.searcher.search(q_or, remaining_k).hits

                for score, doc_address in res_or:
                    if len(candidates) >= top_k:
                        break

                    doc = self.searcher.doc(doc_address)
                    c_id = doc['case_id'][0]
                    slug = doc['slug'][0]

                    if c_id not in seen_ids:
                        candidates.append({
                            "case_id": c_id,
                            "slug": slug,
                            "score": score,
                            "tier": "relaxed_or"
                        })
                        seen_ids.add(c_id)
            except Exception:
                pass

        return candidates
