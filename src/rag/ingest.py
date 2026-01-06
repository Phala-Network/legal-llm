import os
import json
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import glob
from tqdm import tqdm
import re
from dotenv import load_dotenv
import tantivy
import concurrent.futures
import threading
import sys

# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.case_parser import CaseParser
from src.rag.shard_manager import ShardManager
from src.utils.path_utils import normalize_case_path

load_dotenv()


class CaseIngester:
    def __init__(self, data_dir="data", db_path="chroma_db", shard_assignments="data/shard_assignments.json", index_dir="tantivy_index"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.client = OpenAI()
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large"
        )
        self.collections = {}  # Cache loaded collections: name -> collection_obj
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.collection_lock = threading.Lock()

        # Initialize Parser
        self.parser = CaseParser(data_dir=data_dir)

        # Initialize ShardManager
        self.shard_manager = ShardManager(assignments_path=shard_assignments)
        self.shard_assignments_path = shard_assignments

        # Tantivy Index Settings
        self.index_dir = index_dir

    def get_collection(self, collection_name):
        """
        Get or create a specific collection.
        """
        with self.collection_lock:
            if collection_name in self.collections:
                return self.collections[collection_name]

            try:
                col = self.chroma_client.get_collection(name=collection_name)
            except:
                col = self.chroma_client.create_collection(name=collection_name)
                print(f"Created new collection: {collection_name}")

            self.collections[collection_name] = col
            return col

    def load_case_json(self, file_path):
        """Loads the full JSON case file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def chunk_text(self, parsed_data, case_title, decision_date):
        """
        Chunks text by paragraphs with secondary length checks.
        Uses parsed structure (Head Matter + Content Blocks).
        """
        chunks = []
        target_size = 1000
        context_str = f"Case: {case_title} ({decision_date})\n"

        # Combine head matter and content blocks into a single stream of text segments
        # We process them sequentially but keep the "Header" logic in mind if we want to improve semantic chunking later.
        # For now, treat them as a stream of text blocks.

        text_stream = []
        if parsed_data.get("head_matter"):
            text_stream.append(f"HEAD MATTER:\n{parsed_data['head_matter']}")

        if parsed_data.get("content_blocks"):
            text_stream.extend(parsed_data["content_blocks"])

        current_chunk = ""

        for block in text_stream:
            # Further split block by paragraphs if it's large (CaseParser might return large blocks if no headers found)
            # Simple assumption: CaseParser returns logical blocks (paragraphs or sections).
            # But CaseParser right now splits by headers. A section might be huge.
            # So we stick to the paragraph splitting logic WITHIN each block.

            paragraphs = block.split("\n")
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            for p in paragraphs:
                if len(current_chunk) + len(p) < target_size:
                    current_chunk += "\n" + p
                else:
                    if current_chunk:
                        chunks.append(f"{context_str}{current_chunk.strip()}")

                    if len(p) > target_size:
                        # Split huge single paragraph
                        sentences = re.split(r"(?<=[.!?])\s+", p)
                        sub_chunk = ""
                        for s in sentences:
                            if len(sub_chunk) + len(s) < target_size:
                                sub_chunk += " " + s
                            else:
                                chunks.append(f"{context_str}{sub_chunk.strip()}")
                                sub_chunk = s
                        current_chunk = sub_chunk if sub_chunk else ""
                    else:
                        current_chunk = p

        if current_chunk:
            chunks.append(f"{context_str}{current_chunk.strip()}")

        return chunks

    def setup_tantivy(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        # Define Schema
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_unsigned_field("case_id", stored=True)
        schema_builder.add_text_field("slug", stored=True) # Normalized path
        schema = schema_builder.build()

        # Create Index
        index = tantivy.Index(schema, path=self.index_dir)
        return index, index.writer()

    def ingest(self, search_dir: str = None, neighborhoods_path: str = None):
        scan_dir = search_dir or self.data_dir

        # 1. Shard Assignment Generation
        # Only generate if they don't exist yet
        already_loaded = self.shard_manager.load_assignments()

        if neighborhoods_path and os.path.exists(neighborhoods_path) and not already_loaded:
            print(f"Generating shard assignments from {neighborhoods_path}...")
            self.shard_manager.generate_assignments(neighborhoods_path)

        if not self.shard_manager.load_assignments():
            print(f"Error: Shard assignments not found at {self.shard_assignments_path}. Please provide --neighborhoods if this is the first run.")
            return

        print(f"Scanning {scan_dir} ...")
        json_files = glob.glob(
            os.path.join(scan_dir, "**", "json", "*.json"), recursive=True
        )
        # Filter metadata
        json_files = [f for f in json_files if "Metadata" not in f]
        print(f"Found {len(json_files)} case files.")

        # 2. Setup Tantivy
        tantivy_index, tantivy_writer = self.setup_tantivy()

        batches = {}  # "collection_name" -> {ids: [], docs: [], metas: []}
        batch_size = 50
        futures = []

        for json_file in tqdm(json_files):
            try:
                # Determine case ID and assigned shards
                rel_path = os.path.relpath(json_file, self.data_dir)
                case_id_for_shard = normalize_case_path(rel_path)

                target_shards = self.shard_manager.get_shards_for_case(case_id_for_shard)
                if not target_shards:
                    print(f"Warning: No shards assigned for {case_id_for_shard}. Skipping.")
                    continue

                target_collection_names = [f"shard_{s:03d}" for s in target_shards]

                # Extract and parse case data
                case_data = self.load_case_json(json_file)
                jurisdiction = case_data.get("jurisdiction", {}).get("name_long", "Unknown")
                case_id_internal = int(case_data["id"])
                name = case_data.get("name_abbreviation", case_data.get("name", "Unknown"))
                date = case_data.get("decision_date", "Unknown")
                citation = str(case_data.get("citations", [{}])[0].get("cite", ""))

                parsed_structure = self.parser.parse_case_structure(case_data)

                # Add to Tantivy Index
                head_matter = parsed_structure.get("head_matter", "")
                content_blocks = parsed_structure.get("content_blocks", [])
                full_text_for_index = head_matter + "\n" + "\n".join(content_blocks)

                tantivy_writer.add_document(tantivy.Document(
                    title=name,
                    body=full_text_for_index,
                    case_id=case_id_internal,
                    slug=case_id_for_shard
                ))

                # Chunk for Vector DB
                chunks = self.chunk_text(parsed_structure, name, date)

                # Add to batches for each assigned shard
                for col_key in target_collection_names:
                    self.get_collection(col_key) # Ensure exists

                    if col_key not in batches:
                        batches[col_key] = {"ids": [], "docs": [], "metas": []}

                    for i, chunk in enumerate(chunks):
                        doc_id = f"{case_id_internal}_{i}"

                        batches[col_key]["ids"].append(doc_id)
                        batches[col_key]["docs"].append(chunk)
                        batches[col_key]["metas"].append(
                            {
                                "case_id": str(case_id_internal),
                                "path_id": case_id_for_shard,
                                "name": name,
                                "state": jurisdiction,
                                "citation": citation,
                                "file_path": json_file,
                                "chunk_index": i,
                            }
                        )

                        if len(batches[col_key]["ids"]) >= batch_size:
                            batch_copy = batches[col_key]
                            batches[col_key] = {"ids": [], "docs": [], "metas": []}
                            futures.append(
                                self.executor.submit(self._flush_batch, col_key, batch_copy)
                            )

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

        # Final commit and flush
        print("Finishing Tantivy index and remaining vector batches...")
        tantivy_writer.commit()

        for col_key, batch_data in batches.items():
            if batch_data["ids"]:
                futures.append(
                    self.executor.submit(self._flush_batch, col_key, batch_data)
                )

        print("Waiting for pending embeddings...")
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

        print("Ingestion complete.")

    def _flush_batch(self, col_name, batch_data):
        try:
            col = self.chroma_client.get_collection(name=col_name)
            resp = self.client.embeddings.create(
                input=batch_data["docs"], model=self.embedding_model_name
            )
            embeddings = [d.embedding for d in resp.data]

            col.upsert(
                ids=batch_data["ids"],
                documents=batch_data["docs"],
                metadatas=batch_data["metas"],
                embeddings=embeddings,
            )
        except Exception as e:
            print(f"Batch Error for {col_name}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Ingestion: Shard assignments -> Vector Index -> Global Router Index.")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory for ID normalization.")
    parser.add_argument("--search_dir", type=str, default=None, help="Specific subdirectory to scan (for testing).")
    parser.add_argument("--neighborhoods", type=str, default=None, help="Path to case neighborhoods JSON to generate/update assignments.")
    parser.add_argument("--db_path", type=str, default="chroma_db", help="Path to ChromaDB storage.")
    parser.add_argument("--assignments", type=str, default="data/shard_assignments.json", help="Path to shard assignments JSON.")
    parser.add_argument("--index_dir", type=str, default="tantivy_index", help="Path to Tantivy index directory.")

    args = parser.parse_args()

    ingester = CaseIngester(
        data_dir=args.data_dir,
        db_path=args.db_path,
        shard_assignments=args.assignments,
        index_dir=args.index_dir
    )
    ingester.ingest(search_dir=args.search_dir, neighborhoods_path=args.neighborhoods)
