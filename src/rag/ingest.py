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

class CaseIngester:
    def __init__(self, data_dir="data", db_path="chroma_db", collection_name="law_cases"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # Using OpenAI embeddings as requested
        self.client = OpenAI()
        self.embedding_model_name = "openai/text-embedding-3-large"
        
        # Reset or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.embedding_model_name).data[0].embedding

    def load_case(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def extract_and_chunk_html(self, json_path):
        html_path = json_path.replace('/json/', '/html/').replace('.json', '.html')
        
        if not os.path.exists(html_path):
            return []

        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Naive HTML paragraph parsing using regex to avoid heavy deps if possible, 
            # but aiming for <p> content.
            # Convert <br> to newline
            content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
            
            # Find all content inside <p>...</p> tags
            # non-greedy match
            # This captures specific paragraphs
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean tags inside paragraphs
            clean_func = lambda x: re.sub(r'<.*?>', '', x).strip()
            paragraphs = [clean_func(p) for p in paragraphs]
            paragraphs = [p for p in paragraphs if p] # verify non-empty

            # If no p tags found (some files might be raw), fallback to whole text chunking
            if not paragraphs:
                text = re.sub(r'<.*?>', ' ', content).strip()
                if text:
                    paragraphs = [text] # Treat as one giant para (will be subchunked if needed logic added, but for now just Semantic grouping)

            # Semantic Grouping
            # We want chunks around 500-1000 characters.
            chunks = []
            current_chunk = ""
            
            target_size = 800
            
            for p in paragraphs:
                if len(current_chunk) + len(p) < target_size:
                    current_chunk += " " + p
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = p
                    
                    # If a single paragraph is huge > target_size, just add it as is (or split it further if strict)
                    # For legal, sometimes long paras are distinct units. Let's keep them whole if possible unless massive.
                    if len(current_chunk) > target_size * 2:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks

        except Exception as e:
            print(f"Error reading HTML {html_path}: {e}")
            return []

    def ingest(self):
        print(f"Scanning {self.data_dir}...")
        json_files = glob.glob(os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True)
        print(f"Found {len(json_files)} case metadata files.")
        
        batch_size = 50
        ids_batch = []
        docs_batch = []
        metas_batch = []
        
        for json_file in tqdm(json_files):
            try:
                case_meta = self.load_case(json_file)
                state = case_meta.get('jurisdiction', {}).get('name_long', 'Unknown')
                
                chunks = self.extract_and_chunk_html(json_file)
                
                for i, chunk in enumerate(chunks):
                    # Unique ID: CaseID_ChunkIndex
                    doc_id = f"{case_meta['id']}_{i}"
                    
                    ids_batch.append(doc_id)
                    docs_batch.append(chunk)
                    metas_batch.append({
                        "case_id": str(case_meta['id']),
                        "name": case_meta.get('name', ''),
                        "state": state,
                        "citation": str(case_meta.get('citations', [{}])[0].get('cite', '')),
                        "chunk_index": i
                    })
                    
                    if len(ids_batch) >= batch_size:
                        self._add_batch(ids_batch, docs_batch, metas_batch)
                        ids_batch = []
                        docs_batch = []
                        metas_batch = []
                        
            except Exception as e:
                print(f"Skipping {json_file}: {e}")

        # Final batch
        if ids_batch:
            self._add_batch(ids_batch, docs_batch, metas_batch)

    def _add_batch(self, ids, docs, metas):
        try:
            # Batch generate embeddings
            # OpenAI API handles lists of strings (up to 2048 dims usually)
            resp = self.client.embeddings.create(input=docs, model=self.embedding_model_name)
            embeddings = [d.embedding for d in resp.data]
            
            self.collection.upsert( # Use upsert to avoid error on duplicates
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings
            )
        except Exception as e:
            print(f"Batch Error: {e}")

if __name__ == "__main__":
    ingester = CaseIngester()
    ingester.ingest()
