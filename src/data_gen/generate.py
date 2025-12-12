import os
import json
import random
import glob
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import sys

load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.rag.retriever import CaseRetriever

class DataGenerator:
    def __init__(self, data_dir: str = "data", output_file: str = "training_data.jsonl", model: Optional[str] = None):
        self.data_dir = data_dir
        self.output_file = output_file
        self.client = OpenAI()
        self.model = model or os.getenv("GENERATION_MODEL", "openai/gpt-4o")
        
        print("Initializing Real RAG Retriever...")
        import chromadb
        try:
            temp_client = chromadb.PersistentClient(path="chroma_db")
            cols = [c.name for c in temp_client.list_collections() if c.name.startswith("law_cases")]
            target_col = cols[0] if cols else "law_cases"
            self.retriever = CaseRetriever(collection_name=target_col)
        except Exception as e:
            print(f"Warning: Could not initialize retriever ({e}). retrieval capabilities will fail.")
            self.retriever = None

    def _get_case_text(self, json_path: str) -> Dict:
        """Extracts text and metadata from the JSON case file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            opinions = data.get('casebody', {}).get('opinions', [])
            if not opinions:
                return {}
            
            # Combine opinions
            text = "\n\n".join([op.get('text', '') for op in opinions])
            
            return {
                "text": text,
                "id": str(data.get('id', '')),
                "name": data.get('name_abbreviation', data.get('name', 'Unknown Case'))
            }
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            return {}


    def _generate_queries(self, case_info: Dict) -> List[Dict]:
        """
        Step 1: Generate Questions and Search Queries based on the Gold Case.
        We ask the model to generate questions that CAN be answered by this case.
        """
        context_text = case_info.get("text", "")
        case_id = case_info.get("id", "unknown")
        case_name = case_info.get("name", "Unknown Case")

        prompt = f"""
        You are an expert legal data annotator.
        Task: Create 10 diverse training queries based on the provided legal opinion.
        
        The goal is to test a RAG system. Generate questions where the provided case is the CORRECT answer source.
        
        Distribution:
        1. [COMPLEX] (Quantity: 4): Multi-step reasoning questions. 
           - MUST require search. `search_query` should be populated.
        
        2. [SIMPLE] (Quantity: 4): Specific fact retrieval questions.
           - Two should be answerable via Search. `search_query` populated.
           - Two should be a Direct Answer (e.g. general legal concept or definition mentioned in text) with no `search_query`.
        
        3. [NEGATIVE] (Quantity: 2): Unanswerable/Irrelevant questions.
           - One where the assistant searches but finds nothing. `search_query` populated.
           - One where the question is out-of-scope/malformed and rejected immediately with no `search_query`.
        
        Output Structure:
        [
            {{
                "type": "complex",
                "question": "...",
                "search_query": "..."
            }},
            {{
                "type": "simple",
                "question": "...",
                "search_query": "..."  // or null
            }},
            {{
                "type": "negative",
                "question": "...",
                "search_query": "..." // or null
            }}
        ]

        Case Metadata:
        ID: {case_id}
        Name: {case_name}

        Case Text (Truncated):
        {context_text[:15000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Output valid JSON list only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.7
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list): return v
            return data if isinstance(data, list) else []
            
        except Exception as e:
            print(f"Query Gen error: {e}")
            return []

    def _generate_answers_batch(self, items: List[Dict], gold_text: str) -> List[Dict]:
        """
        Step 3 (Batched): Generate final answers for multiple questions in one go.
        """
        # Construct a combined prompt
        task_prompt = "You are a legal AI assistant.\nContext (Gold Case):\n" + gold_text[:15000] + "\n\nTask: Answer the following questions based on their specific search results (if any) or the general context.\n\n"
        
        for i, item in enumerate(items):
            q_text = item['q_item']['question']
            results = item.get('retrieved_context_str', "No search performed.")
            task_prompt += f"--- ITEM {i} ---\nQuestion: {q_text}\nSearch Results Context:\n{results}\n\n"
            
        task_prompt += """
        IMPORTANT INSTRUCTIONS:
        1. If you use information from a Search Result, you MUST cite it at the end of the sentence or paragraph.
        2. Citation Format: [Case Name](/cases/CaseID)
           - Example: ...as established in [Smith v. Jones](/cases/12345).
           - The CaseID and Name are found in the "[Result X] Name (ID: ...)" header.
        3. If "No search performed" (Direct Answer), cite the Gold Case context if applicable, or state general knowledge.
        
        Output a JSON list of objects, one for each Item, in order:
        [
            { "thought": "...", "answer": "..." },
            ...
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model, # Use the strong model
                messages=[
                    {"role": "system", "content": "Output valid JSON list only."},
                    {"role": "user", "content": task_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.3
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Normalize
            if isinstance(data, dict):
                 for k,v in data.items():
                     if isinstance(v, list): return v
            return data if isinstance(data, list) else []
            
        except Exception as e:
            print(f"Batch Answer Gen error: {e}")
            return []

    def run(self, num_samples: int = 10):
        json_files = glob.glob(os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True)
        if not json_files:
            print("No data files found.")
            return

        selected_files = random.sample(json_files, min(num_samples, len(json_files)))
        
        valid_count = 0
        print(f"Starting generation: {len(selected_files)} files selected. Output: {self.output_file}")
        
        if os.path.exists(self.output_file):
            print(f"Appending to existing file: {self.output_file}")
            
        for json_file in tqdm(selected_files, desc="Processing Cases"):
            case_info = self._get_case_text(json_file)
            if not case_info or len(case_info.get("text", "")) < 500:
                continue

            # Step 1: Generate Queries
            query_batch = self._generate_queries(case_info)
            
            # Step 2: Prepare Batch Implementation
            items_to_process = []
            
            for q_item in query_batch:
                search_query = q_item.get("search_query")
                
                # Retrieval
                retrieved_docs = []
                context_str_final = ""
                
                if search_query and self.retriever:
                     retrieved_docs = self.retriever.retrieve(search_query, k=4)
                     for i, doc in enumerate(retrieved_docs):
                        real_id = doc["id"].split('_')[0] if "_" in doc["id"] else doc["id"]
                        name = doc.get("metadata", {}).get("name", "Unknown Case")
                        context_str_final += f"[Result {i+1}] {name} (ID: {real_id})\n{doc['text'][:800]}...\n\n"
                
                items_to_process.append({
                    "q_item": q_item,
                    "search_query": search_query,
                    "retrieved_context_str": context_str_final
                })
            
            # Step 3: Batched Answer Generation
            if not items_to_process: continue
            
            answers = self._generate_answers_batch(items_to_process, case_info['text'])
            
            # Match answers back
            if len(answers) != len(items_to_process):
                print(f"Warning: Batch size mismatch (Sent {len(items_to_process)}, Got {len(answers)}). Skipping batch.")
                continue

            for i, item in enumerate(items_to_process):
                try:
                    q_item = item['q_item']
                    ans_data = answers[i]
                    search_query = item['search_query']
                    context_str_final = item['retrieved_context_str']
                    
                    # Construct Final Entry
                    messages = [{"role": "user", "content": q_item["question"]}]
                    
                    if search_query:
                        messages.append({"role": "assistant", "content": f"<thought>{ans_data.get('thought', 'Thinking...')}</thought>\n<search>{search_query}</search>"})
                        messages.append({"role": "user", "content": f"Search Results:\n{context_str_final}\n"})
                    
                    messages.append({"role": "assistant", "content": ans_data.get("answer", "I cannot answer.")})
                    
                    entry = {"messages": messages}
                    
                    with open(self.output_file, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(entry) + "\n")
                    valid_count += 1
                except Exception as e:
                    print(f"Skipping entry in batch: {e}")

        print(f"Data Generation Complete. Generated {valid_count} examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
