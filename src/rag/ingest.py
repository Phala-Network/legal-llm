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

class CaseIngester:
    def __init__(self, data_dir="data", db_path="chroma_db"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.client = OpenAI()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "openai/text-embedding-3-large")
        self.collections = {} # Cache loaded collections: name -> collection_obj
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.collection_lock = threading.Lock()

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

    def extract_text_from_json(self, case_data):
        """Extracts text from the 'casebody' -> 'opinions' structure."""
        try:
            opinions = case_data.get('casebody', {}).get('opinions', [])
            if not opinions:
                return ""
            
            # Combine all opinions (majority, dissenting, etc.)
            full_text = "\n\n".join([op.get('text', '') for op in opinions])
            return full_text
        except Exception as e:
            print(f"Error extracting text from JSON: {e}")
            return ""

    def chunk_text(self, text, case_title, decision_date):
        """
        Chunks text by paragraphs with secondary length checks.
        Prepends context (Case Name, Date) to the valid chunks.
        """
        if not text:
            return []

        # 1. Split by Paragraphs
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""
        target_size = 1000
        
        # Context String
        # e.g. "Context: United Riggers v. Coast Iron (2015-11-23)\n"
        context_str = f"Case: {case_title} ({decision_date})\n"

        for p in paragraphs:
            # Check potential size
            if len(current_chunk) + len(p) < target_size:
                current_chunk += "\n" + p
            else:
                # Flush current chunk if it exists
                if current_chunk:
                    chunks.append(f"{context_str}{current_chunk.strip()}")
                
                # Handle the new paragraph
                if len(p) > target_size:
                    # If single paragraph is HUGE, force split by sentences
                    # Simple heuristic split on ". " to avoid chopping words
                    # Better would be nltk/spacy but keeping dependencies low
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_chunk = ""
                    for s in sentences:
                        if len(sub_chunk) + len(s) < target_size:
                            sub_chunk += " " + s
                        else:
                            chunks.append(f"{context_str}{sub_chunk.strip()}")
                            sub_chunk = s
                    if sub_chunk:
                        current_chunk = sub_chunk # Carry over remainder
                    else:
                        current_chunk = ""
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

                text = self.extract_text_from_json(case_data)
                
                # Chunking
                chunks = self.chunk_text(text, name, date)
                
                # Get relevant collection object
                # We normalize name to get the key.
                safe_state_name = re.sub(r'[^a-zA-Z0-9]', '_', jurisdiction.lower())
                col_key = f"law_cases_{safe_state_name}"
                
                # Pre-warm collection safely in main thread
                # This ensures we don't have race conditions on creation inside the thread
                self.get_collection(jurisdiction) 

                if col_key not in batches:
                    batches[col_key] = {"ids": [], "docs": [], "metas": []}
                
                for i, chunk in enumerate(chunks):
                    # Unique ID
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

                    # Check Batch Size for this collection
                    if len(batches[col_key]["ids"]) >= batch_size:
                        # Submit to executor
                        # Must copy the data structure because batches[col_key] will be reset
                        batch_copy = batches[col_key]
                        batches[col_key] = {"ids": [], "docs": [], "metas": []}
                        
                        futures.append(self.executor.submit(self._flush_batch, col_key, batch_copy))

            except Exception as e:
                print(f"Skipping {json_file}: {e}")

        # Final flush
        for col_key, batch_data in batches.items():
            if batch_data["ids"]:
                futures.append(self.executor.submit(self._flush_batch, col_key, batch_data))

        # Wait for all
        print("Waiting for pending embeddings...")
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass # exceptions are caught in _flush_batch but we could inspect f.result()
            
        print("Ingestion complete.")

    def _flush_batch(self, col_name, batch_data):
        try:
            # Route to correct collection 
            # We used get_collection in main thread, so it should be cached or safe to get.
            # We use the lock just in case if we access the cache.
            # Actually, we can just call self.chroma_client.get_collection(name=col_name) directly
            # since we know it exists. But safe to use the helper with lock.
            
            # Since we modify self.collections in get_collection, let's just use the client directly for speed/safety
            # assuming it was created. Or simpler: use helper with lock.
            
            # However, for massive parallelism, we might want to avoid locking on *read* of the cache if possible.
            # But the lock is fine for 5 workers.
            
            # col = self.get_collection( ... wait we only have the safe name here ... )
            # We can't reverse engineer the 'State Name' from 'law_cases_state_name' reliably if we want the pretty name,
            # BUT get_collection only uses the state name to generate the safe name.
            # If we already have the safe name (col_name), we can just get it from chroma client.
            
            col = self.chroma_client.get_collection(name=col_name)

            # Generate Embeddings
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
