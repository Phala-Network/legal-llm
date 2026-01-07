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
from src.utils.batch_utils import BatchManager

load_dotenv()


class CaseIngester:
    def __init__(
        self,
        data_dir="data",
        db_path="chroma_db",
        shard_assignments="data/shard_assignments.json",
        index_dir="tantivy_index",
        batch_dir=".",
    ):
        self.data_dir = data_dir
        self.db_path = db_path
        self.index_dir = index_dir
        self.batch_dir = batch_dir
        os.makedirs(self.batch_dir, exist_ok=True)

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
        self.batch_manager = BatchManager(self.client)

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
        schema_builder.add_text_field("slug", stored=True)  # Normalized path
        schema = schema_builder.build()

        # Create Index
        index = tantivy.Index(schema, path=self.index_dir)
        return index, index.writer()

    def ingest(
        self,
        search_dir: str = None,
        neighborhoods_path: str = None,
        batch_mode: bool = True,
        tantivy_only: bool = False,
    ):
        scan_dir = search_dir or self.data_dir

        # 1. Shard Assignment Generation
        # Only generate if they don't exist yet
        print("Loading shard assignments...")
        already_loaded = self.shard_manager.load_assignments()

        if (
            neighborhoods_path
            and os.path.exists(neighborhoods_path)
            and not already_loaded
        ):
            print(f"Generating shard assignments from {neighborhoods_path}...")
            self.shard_manager.generate_assignments(neighborhoods_path)

        if not self.shard_manager.load_assignments():
            print(
                f"Error: Shard assignments not found at {self.shard_assignments_path}. Please provide --neighborhoods if this is the first run."
            )
            return

        print(f"Scanning {scan_dir} ...")
        json_files = glob.glob(
            os.path.join(scan_dir, "**", "json", "*.json"), recursive=True
        )
        # Filter metadata
        json_files = [f for f in json_files if "Metadata" not in f]
        print(f"Found {len(json_files)} case files.")

        if tantivy_only:
            print("Tantivy Only Mode: Building search index...")
            tantivy_index, tantivy_writer = self.setup_tantivy()
            for json_file in tqdm(json_files, desc="Indexing Tantivy"):
                try:
                    rel_path = os.path.relpath(json_file, self.data_dir)
                    case_id_for_shard = normalize_case_path(rel_path)
                    case_data = self.load_case_json(json_file)
                    case_id_internal = int(case_data["id"])
                    name = case_data.get(
                        "name_abbreviation", case_data.get("name", "Unknown")
                    )
                    parsed_structure = self.parser.parse_case_structure(case_data)
                    head_matter = parsed_structure.get("head_matter", "")
                    content_blocks = parsed_structure.get("content_blocks", [])
                    full_text = head_matter + "\n" + "\n".join(content_blocks)

                    tantivy_writer.add_document(
                        tantivy.Document(
                            title=name,
                            body=full_text,
                            case_id=case_id_internal,
                            slug=case_id_for_shard,
                        )
                    )
                except Exception as e:
                    print(f"Skipping {json_file} for Tantivy: {e}")
            tantivy_writer.commit()
            print("Tantivy index built successfully.")
            return

        if batch_mode:
            self.submit_batch_ingestion(json_files)
            return

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

                target_shards = self.shard_manager.get_shards_for_case(
                    case_id_for_shard
                )
                if not target_shards:
                    print(
                        f"Warning: No shards assigned for {case_id_for_shard}. Skipping."
                    )
                    continue

                target_collection_names = [f"shard_{s:03d}" for s in target_shards]

                # Extract and parse case data
                case_data = self.load_case_json(json_file)
                jurisdiction = case_data.get("jurisdiction", {}).get(
                    "name_long", "Unknown"
                )
                case_id_internal = int(case_data["id"])
                name = case_data.get(
                    "name_abbreviation", case_data.get("name", "Unknown")
                )
                date = case_data.get("decision_date", "Unknown")
                citation = str(case_data.get("citations", [{}])[0].get("cite", ""))

                parsed_structure = self.parser.parse_case_structure(case_data)

                # Add to Tantivy Index
                head_matter = parsed_structure.get("head_matter", "")
                content_blocks = parsed_structure.get("content_blocks", [])
                full_text_for_index = head_matter + "\n" + "\n".join(content_blocks)

                tantivy_writer.add_document(
                    tantivy.Document(
                        title=name,
                        body=full_text_for_index,
                        case_id=case_id_internal,
                        slug=case_id_for_shard,
                    )
                )

                # Chunk for Vector DB
                chunks = self.chunk_text(parsed_structure, name, date)

                # Add to batches for each assigned shard
                for col_key in target_collection_names:
                    self.get_collection(col_key)  # Ensure exists

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
                                self.executor.submit(
                                    self._flush_batch, col_key, batch_copy
                                )
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

    def submit_batch_ingestion(self, json_files):
        print("Preparing Batch API submission...")
        meta_file = os.path.join(self.batch_dir, "ingest_meta.jsonl")

        requests = []
        metadata_map = []

        # Max requests per batch file (limit is 50,000)
        BATCH_LIMIT = getattr(self, "BATCH_LIMIT", 50000)
        batch_files = []

        # We need access to shard assignments to know where things go ultimately,
        # but for batch processing, we just need the embeddings first.
        # We can store the shard destination in the metadata map.

        count = 0
        file_idx = 1

        for json_file in tqdm(json_files):
            try:
                rel_path = os.path.relpath(json_file, self.data_dir)
                case_id_path = normalize_case_path(rel_path)

                # Check shards
                target_shards = self.shard_manager.get_shards_for_case(case_id_path)
                if not target_shards:
                    continue

                target_collections = [f"shard_{s:03d}" for s in target_shards]

                case_data = self.load_case_json(json_file)
                parsed = self.parser.parse_case_structure(case_data)

                name = case_data.get(
                    "name_abbreviation", case_data.get("name", "Unknown")
                )
                date = case_data.get("decision_date", "Unknown")
                citation = str(case_data.get("citations", [{}])[0].get("cite", ""))
                jurisdiction = case_data.get("jurisdiction", {}).get(
                    "name_long", "Unknown"
                )
                cid_int = int(case_data["id"])

                chunks = self.chunk_text(parsed, name, date)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{cid_int}_{i}"
                    custom_id = f"req-{doc_id}"

                    # Store metadata required for ingestion later
                    meta_entry = {
                        "custom_id": custom_id,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk,
                        "target_collections": target_collections,
                        "metadata": {
                            "case_id": str(cid_int),
                            "path_id": case_id_path,
                            "name": name,
                            "state": jurisdiction,
                            "citation": citation,
                            "file_path": json_file,
                            "chunk_index": i,
                        },
                    }
                    metadata_map.append(meta_entry)

                    # Prepare Batch Request
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {"model": self.embedding_model_name, "input": chunk},
                    }
                    requests.append(request)
                    count += 1

                    if len(requests) >= BATCH_LIMIT:
                        current_batch_file = os.path.join(
                            self.batch_dir, f"ingest_batch_input_{file_idx}.jsonl"
                        )

                        # Write exactly BATCH_LIMIT
                        self._write_batch_file(
                            current_batch_file, requests[:BATCH_LIMIT]
                        )
                        batch_files.append(current_batch_file)
                        print(f"Prepared {current_batch_file} ({BATCH_LIMIT} requests)")

                        # Keep the rest (if any)
                        requests = requests[BATCH_LIMIT:]
                        file_idx += 1

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

        # Flush remaining
        if requests:
            current_batch_file = os.path.join(
                self.batch_dir, f"ingest_batch_input_{file_idx}.jsonl"
            )
            self._write_batch_file(current_batch_file, requests)
            batch_files.append(current_batch_file)
            print(f"Prepared {current_batch_file} ({len(requests)} requests)")

        # Save metadata map locally (crucial for resume step)
        print(f"Saving metadata map to {meta_file}...")
        with open(meta_file, "w") as f:
            for item in metadata_map:
                f.write(json.dumps(item) + "\n")

        # Submit Batches
        print("\nSubmitting Batches to OpenAI...")
        batch_ids = []
        for f_path in batch_files:
            fid = self.batch_manager.upload_file(f_path)
            bid = self.batch_manager.create_batch(
                fid, description=f"Ingest {f_path}", endpoint="/v1/embeddings"
            )
            batch_ids.append(bid)

        print("\n" + "=" * 50)
        print("BATCH SUBMISSION COMPLETE")
        print("Batch IDs: " + ", ".join(batch_ids))
        print("Polling for completion...")
        print("=" * 50 + "\n")

        self._poll_and_ingest_batch(",".join(batch_ids))

    def _write_batch_file(self, filename, requests):
        with open(filename, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

    def _poll_and_ingest_batch(self, batch_ids_str):
        batch_ids = [b.strip() for b in batch_ids_str.split(",")]
        meta_file = os.path.join(self.batch_dir, "ingest_meta.jsonl")

        if not os.path.exists(meta_file):
            print(
                f"Error: Metadata file {meta_file} not found. Cannot match embeddings to documents."
            )
            return

        print("Loading metadata map...")
        # Load meta into a dict for O(1) lookup: custom_id -> meta_entry
        meta_lookup = {}
        with open(meta_file, "r") as f:
            for line in f:
                item = json.loads(line)
                meta_lookup[item["custom_id"]] = item

        print(f"Loaded {len(meta_lookup)} metadata entries.")

        # Process each batch
        for bid in batch_ids:
            try:
                print(f"\nProcessing Batch {bid}...")
                out_fid = self.batch_manager.wait_for_batch(bid)
                local_file = os.path.join(self.batch_dir, f"batch_output_{bid}.jsonl")
                self.batch_manager.download_file(out_fid, local_file)

                # Ingest results
                self._process_batch_results(bid, local_file, meta_lookup)

            except Exception as e:
                print(f"Error processing batch {bid}: {e}")

        print("\nBatch Ingestion Complete.")

    def _process_batch_results(self, batch_id, local_file, meta_lookup):
        print(f"Ingesting embeddings from {local_file}...")
        # We need to buffer writes to Chroma to differenct collections
        # buffer structure: collection_name -> {ids: [], embeddings: [], metadatas: [], documents: []}
        buffer = {}
        processed_count = 0

        # We skip JSONL loading here for memory, but for progress bar we might need length
        # Open once to count if possible, or just line by line
        with open(local_file, "r") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Ingesting {batch_id}", unit="emb"):
            try:
                res = json.loads(line)
                cid = res["custom_id"]

                if res.get("error"):
                    print(f"Error in result {cid}: {res['error']}")
                    continue

                if cid not in meta_lookup:
                    print(f"Warning: Unknown custom_id {cid}")
                    continue

                meta_entry = meta_lookup[cid]
                embedding = res["response"]["body"]["data"][0]["embedding"]

                # Each chunk might go to multiple shard collections
                for col_name in meta_entry["target_collections"]:
                    if col_name not in buffer:
                        buffer[col_name] = {
                            "ids": [],
                            "embeddings": [],
                            "metadatas": [],
                            "documents": [],
                        }

                    buffer[col_name]["ids"].append(meta_entry["doc_id"])
                    buffer[col_name]["embeddings"].append(embedding)
                    buffer[col_name]["metadatas"].append(meta_entry["metadata"])
                    buffer[col_name]["documents"].append(meta_entry["text"])

                processed_count += 1

                if processed_count % 1000 == 0:
                    self._flush_chroma_buffer(buffer)
                    buffer = (
                        {}
                    )  # Reset specific collections? No, reset all to keep memory low
            except Exception as e:
                print(f"Error processing line in {local_file}: {e}")

        # Final flush
        self._flush_chroma_buffer(buffer)
        print(f"Ingested {processed_count} embeddings.")

    def _flush_chroma_buffer(self, buffer):
        for col_name, data in buffer.items():
            if not data["ids"]:
                continue
            try:
                col = self.get_collection(col_name)
                col.upsert(
                    ids=data["ids"],
                    embeddings=data["embeddings"],
                    metadatas=data["metadatas"],
                    documents=data["documents"],
                )
            except Exception as e:
                print(f"Error flushing to {col_name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Ingestion: Shard assignments -> Vector Index -> Global Router Index."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory for ID normalization.",
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        default=None,
        help="Specific subdirectory to scan (for testing).",
    )
    parser.add_argument(
        "--neighborhoods",
        type=str,
        default=None,
        help="Path to case neighborhoods JSON to generate/update assignments.",
    )
    parser.add_argument(
        "--db_path", type=str, default="chroma_db", help="Path to ChromaDB storage."
    )
    parser.add_argument(
        "--assignments",
        type=str,
        default="data/shard_assignments.json",
        help="Path to shard assignments JSON.",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="tantivy_index",
        help="Path to Tantivy index directory.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use legacy synchronous ingestion (no Batch API).",
    )
    parser.add_argument(
        "--tantivy_only",
        action="store_true",
        help="Build only the Tantivy index, skip embeddings.",
    )
    parser.add_argument(
        "--batch_dir",
        type=str,
        default=".",
        help="Directory to store batch input/output files.",
    )

    args = parser.parse_args()

    ingester = CaseIngester(
        data_dir=args.data_dir,
        db_path=args.db_path,
        shard_assignments=args.assignments,
        index_dir=args.index_dir,
        batch_dir=args.batch_dir,
    )
    ingester.ingest(
        search_dir=args.search_dir,
        neighborhoods_path=args.neighborhoods,
        batch_mode=not args.sync,
        tantivy_only=args.tantivy_only,
    )
