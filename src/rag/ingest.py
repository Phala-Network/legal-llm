import os
import json
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import glob
from tqdm import tqdm
import re
from dotenv import load_dotenv

load_dotenv()

import concurrent.futures
import threading
import sys

# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.rag.case_parser import CaseParser

class CaseIngester:
    def __init__(self, data_dir="data", db_path="chroma_db"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.client = OpenAI()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large")
        self.collections = {} # Cache loaded collections: name -> collection_obj
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.collection_lock = threading.Lock()

        # Initialize Parser
        self.parser = CaseParser(data_dir=data_dir)

    def get_collection(self, state_name):
        """
        Get or create a collection for a specific state.
        Normalizes state name to safe collection name string.
        """
        # Normalize: California -> law_cases_california
        # "New York" -> law_cases_new_york
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', state_name.lower())
        col_name = f"law_cases_{safe_name}"

        with self.collection_lock:
            if col_name in self.collections:
                return self.collections[col_name]

            try:
                col = self.chroma_client.get_collection(name=col_name)
                # print(f"Loaded existing collection: {col_name}")
            except:
                col = self.chroma_client.create_collection(name=col_name)
                print(f"Created new collection: {col_name}")

            self.collections[col_name] = col
            return col

    def load_case_json(self, file_path):
        """Loads the full JSON case file."""
        with open(file_path, 'r') as f:
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
        if parsed_data.get('head_matter'):
             text_stream.append(f"HEAD MATTER:\n{parsed_data['head_matter']}")

        if parsed_data.get('content_blocks'):
            text_stream.extend(parsed_data['content_blocks'])

        current_chunk = ""

        for block in text_stream:
            # Further split block by paragraphs if it's large (CaseParser might return large blocks if no headers found)
            # Simple assumption: CaseParser returns logical blocks (paragraphs or sections).
            # But CaseParser right now splits by headers. A section might be huge.
            # So we stick to the paragraph splitting logic WITHIN each block.

            paragraphs = block.split('\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

            for p in paragraphs:
                if len(current_chunk) + len(p) < target_size:
                    current_chunk += "\n" + p
                else:
                    if current_chunk:
                        chunks.append(f"{context_str}{current_chunk.strip()}")

                    if len(p) > target_size:
                        # Split huge single paragraph
                        sentences = re.split(r'(?<=[.!?])\s+', p)
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

    def ingest(self):
        print(f"Scanning {self.data_dir}...")
        json_files = glob.glob(os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True)
        print(f"Found {len(json_files)} case files.")

        batches = {} # "collection_name" -> {ids: [], docs: [], metas: []}
        batch_size = 50
        futures = []

        for json_file in tqdm(json_files):
            try:
                case_data = self.load_case_json(json_file)

                # Metadata extraction
                jurisdiction = case_data.get('jurisdiction', {}).get('name_long', 'Unknown')
                case_id = str(case_data['id'])
                name = case_data.get('name_abbreviation', case_data.get('name', 'Unknown'))
                date = case_data.get('decision_date', 'Unknown')
                citation = str(case_data.get('citations', [{}])[0].get('cite', ''))

                # USE NEW PARSER
                parsed_structure = self.parser.parse_case_structure(case_data)

                # Chunking
                chunks = self.chunk_text(parsed_structure, name, date)

                # Get relevant collection object
                safe_state_name = re.sub(r'[^a-zA-Z0-9]', '_', jurisdiction.lower())
                col_key = f"law_cases_{safe_state_name}"

                self.get_collection(jurisdiction)

                if col_key not in batches:
                    batches[col_key] = {"ids": [], "docs": [], "metas": []}

                for i, chunk in enumerate(chunks):
                    doc_id = f"{case_id}_{i}"

                    batches[col_key]["ids"].append(doc_id)
                    batches[col_key]["docs"].append(chunk)
                    batches[col_key]["metas"].append({
                        "case_id": case_id,
                        "name": name,
                        "state": jurisdiction,
                        "citation": citation,
                        "chunk_index": i
                    })

                    if len(batches[col_key]["ids"]) >= batch_size:
                        batch_copy = batches[col_key]
                        batches[col_key] = {"ids": [], "docs": [], "metas": []}
                        futures.append(self.executor.submit(self._flush_batch, col_key, batch_copy))

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

        for col_key, batch_data in batches.items():
            if batch_data["ids"]:
                futures.append(self.executor.submit(self._flush_batch, col_key, batch_data))

        print("Waiting for pending embeddings...")
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

        print("Ingestion complete.")

    def _flush_batch(self, col_name, batch_data):
        try:
            col = self.chroma_client.get_collection(name=col_name)
            resp = self.client.embeddings.create(input=batch_data["docs"], model=self.embedding_model_name)
            embeddings = [d.embedding for d in resp.data]

            col.upsert(
                ids=batch_data["ids"],
                documents=batch_data["docs"],
                metadatas=batch_data["metas"],
                embeddings=embeddings
            )
        except Exception as e:
            print(f"Batch Error for {col_name}: {e}")

if __name__ == "__main__":
    ingester = CaseIngester()
    ingester.ingest()
