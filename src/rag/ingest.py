import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import glob
from tqdm import tqdm

class CaseIngester:
    def __init__(self, data_dir="data", db_path="chroma_db", collection_name="law_cases"):
        self.data_dir = data_dir
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        # Using a lightweight model for embeddings as per plan
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

    def load_case(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def extract_text(self, case_data):
        """
        Extracts relevant text from the case JSON. 
        Currently concatenates case name and analysis/summary if available.
        A real implementation might need to parse HTML content if 'full_text' isn't readily available in plain text.
        Looking at the sample 0001-01.json, the actual text isn't directly in the JSON. 
        It seems the JSON has metadata and the text might be in the corresponding HTML file or requires parsing.
        
        However, the user mentioned: "I put some sample data ... unzip the volumn and learn about the data format."
        Let's check if there is a plain text field. 
        The sample JSON `0001-01.json` has `citations`, `court`, `jurisdiction`, `cites_to` etc.
        But NO body text.
        
        Wait, the unzip output showed:
        inflating: data/196/json/0001-01.json  
        inflating: data/196/html/0001-01.html

        So the text is in the HTML files!
        I need to read the corresponding HTML file.
        """
        
        # Simplified for now: specific logic to find HTML file
        # The JSON path is like data/196/json/0001-01.json
        # The HTML path is like data/196/html/0001-01.html
        
        json_path = case_data.get('file_path_local') # I will inject this when calling
        if not json_path:
             return ""
             
        html_path = json_path.replace('/json/', '/html/').replace('.json', '.html')
        
        if os.path.exists(html_path):
            try:
                # Basic text extraction from HTML - for a robust app we might use BeautifulSoup or unstructured
                # For this task, I'll use a simple approach or assuming 'unstructured' library was installed for this.
                # Let's try to just read text for now to keep it simple and dependency-light if possible, 
                # but I installed `unstructured` so I should use it or `beautifulsoup4`.
                # I'll use simple string parsing/regex if bs4 isn't strictly requested, 
                # but likely `BeautifulSoup` is better.
                # Since I didn't explicitly install `beautifulsoup4` (it might be in `unstructured` deps or not),
                # I'll check if I can shell out to it or if it's there. 
                # Actually I installed `unstructured` which is heavy. 
                # Let's stick to simple reading if possible, or use `lxml` if present.
                
                # Let's fallback to rudimentary stripping of tags for now to ensure it works without complex deps if they fail.
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    
                from xml.etree import ElementTree as ET
                # HTML might be malformed, so regex is safer for "just get text"
                import re
                clean = re.compile('<.*?>')
                text = re.sub(clean, ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            except Exception as e:
                print(f"Error reading HTML {html_path}: {e}")
                return ""
        return ""

    def ingest(self):
        print(f"Scanning {self.data_dir}...")
        # Recursively find all JSON files
        json_files = glob.glob(os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True)
        
        print(f"Found {len(json_files)} case metadata files.")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        batch_size = 50
        
        for i, json_file in tqdm(enumerate(json_files), total=len(json_files)):
            try:
                case_meta = self.load_case(json_file)
                case_meta['file_path_local'] = json_file
                
                # Grouping Logic:
                # The Task says: "I plan to group the data by state."
                # The JSON has `jurisdiction`: { "name_long": "California", ... }
                state = case_meta.get('jurisdiction', {}).get('name_long', 'Unknown')
                
                # Filter: The user said "group the data by state".
                # For now, we index everything, but add state to metadata so we can filter at retrieval time.
                
                text_content = self.extract_text(case_meta)
                
                if not text_content:
                    continue
                
                # Chunking could be added here, but for simplicity of this "Assistant" MVP, 
                # we'll index the whole text or the first N chars if it's huge. 
                # Embedding models have limits (e.g. 512 tokens).
                # To do RAG effectively, we SHOULD chunk.
                # Let's do a simple sliding window chunking.
                
                chunks = self.chunk_text(text_content)
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    doc_id = f"{case_meta['id']}_{chunk_idx}"
                    
                    ids.append(doc_id)
                    documents.append(chunk_text)
                    metadatas.append({
                        "case_id": str(case_meta['id']),
                        "name": case_meta.get('name', ''),
                        "state": state,
                        "citation": str(case_meta.get('citations', [{}])[0].get('cite', '')),
                        "chunk_index": chunk_idx
                    })
            except Exception as e:
                print(f"Skipping {json_file}: {e}")
                
            # Batch add
            if len(documents) >= batch_size:
                self._add_batch(ids, documents, metadatas)
                ids = []
                documents = []
                metadatas = []
                embeddings = []

        # Final batch
        if documents:
            self._add_batch(ids, documents, metadatas)

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        # Very naive character chunking
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def _add_batch(self, ids, documents, metadatas):
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

if __name__ == "__main__":
    ingester = CaseIngester()
    ingester.ingest()
