import os
import json
import random
import glob
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DataGenerator:
    def __init__(self, data_dir: str = "data", output_file: str = "training_data.jsonl", model: Optional[str] = None):
        self.data_dir = data_dir
        self.output_file = output_file
        self.client = OpenAI()
        self.model = model or os.getenv("GENERATION_MODEL", "openai/gpt-5.1")

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

    def _generate_examples(self, case_info: Dict) -> List[Dict]:
        """
        Generates a mix of training examples from the text in a single pass.
        Distribution:
        - 2 Complex "Agentic RAG" examples
        - 2 Simple QA examples
        - 1 Negative example
        """
        context_text = case_info.get("text", "")
        case_id = case_info.get("id", "unknown")
        case_name = case_info.get("name", "Unknown Case")

        prompt = f"""
        You are an expert legal data annotator.
        Task: Create 5 diverse training examples based on the provided legal opinion.
        
        The goal is to create a robust dataset for a Legal AI Assistant.
        
        Distribution of examples to generate:
        1. [COMPLEX] (Quantity: 2): Multi-step reasoning.
           - User asks a complex question.
           - Assistant thinks, searches, and answers.
           - IMPORTANT: The assistant MUST cite the case in the final answer using the format: [{case_name}](/cases/{case_id})
        2. [SIMPLE] (Quantity: 2): Direct fact retrieval.
           - User asks a specific question.
           - Assistant answers directly and concisely.
           - IMPORTANT: The assistant MUST cite the case in the answer using the format: [{case_name}](/cases/{case_id})
        3. [NEGATIVE] (Quantity: 1): Unanswerable question.
           - User asks a plausible-sounding legal question that CANNOT be answered from this specific text.
           - Assistant politely refuses, stating the information is not in the context.
        
        Structure for [COMPLEX] Example:
        {{
            "type": "complex",
            "question": "...",
            "thought": "Step-by-step reasoning...",
            "search_query": "Keywords for search...",
            "relevant_context": "Verbatim text snippet...",
            "answer": "Final detailed answer with citation [{case_name}](/cases/{case_id})..."
        }}

        Structure for [SIMPLE] Example:
        {{
            "type": "simple",
            "question": "...",
            "answer": "Direct answer with citation [{case_name}](/cases/{case_id})..."
        }}
        
        Structure for [NEGATIVE] Example:
        {{
            "type": "negative",
            "question": "...",
            "answer": "I cannot answer this question based on the provided text, as it does not discuss [task/topic]."
        }}

        Case Metadata:
        ID: {case_id}
        Name: {case_name}

        Case Text (Truncated):
        {context_text[:15000]}

        Output format: A pure JSON list of objects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful and strict data annotation assistant. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.7
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Normalize to list
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Handle {"examples": [...]} or similar wrapper
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
            return []
            
        except Exception as e:
            print(f"Generation error: {e}")
            return []

    def run(self, num_samples: int = 10):
        json_files = glob.glob(os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True)
        if not json_files:
            print("No data files found.")
            return

        selected_files = random.sample(json_files, min(num_samples, len(json_files)))
        
        valid_count = 0
        print(f"Starting generation: {len(selected_files)} files selected. Output: {self.output_file}")
        
        # Check if file exists to determine if we are appending
        if os.path.exists(self.output_file):
            print(f"Appending to existing file: {self.output_file}")
            
        for json_file in tqdm(selected_files, desc="Generating Data"):
            case_info = self._get_case_text(json_file)
            if not case_info or len(case_info.get("text", "")) < 500:
                continue

            examples = self._generate_examples(case_info)
            
            for ex in examples:
                try:
                    entry = None
                    ex_type = ex.get("type", "simple")

                    if ex_type == "complex":
                        if not all(k in ex for k in ["question", "thought", "search_query", "relevant_context", "answer"]):
                            continue
                        entry = {
                            "messages": [
                                {"role": "user", "content": ex["question"]},
                                {"role": "assistant", "content": f"<thought>{ex['thought']}</thought>\n<search>{ex['search_query']}</search>"},
                                # Inject metadata header into search/context result to match inference format
                                {"role": "user", "content": f"Search Results:\n[Result 1] {case_info['name']} (ID: {case_info['id']})\n{ex['relevant_context']}\n"},
                                {"role": "assistant", "content": ex["answer"]}
                            ]
                        }
                    
                    elif ex_type == "simple":
                        if not all(k in ex for k in ["question", "answer"]):
                            continue
                        entry = {
                            "messages": [
                                {"role": "user", "content": ex["question"]},
                                {"role": "assistant", "content": ex["answer"]}
                            ]
                        }

                    elif ex_type == "negative":
                        if not all(k in ex for k in ["question", "answer"]):
                            continue
                        entry = {
                            "messages": [
                                {"role": "user", "content": ex["question"]},
                                {"role": "assistant", "content": ex["answer"]}
                            ]
                        }

                    if entry:
                        with open(self.output_file, 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(entry) + "\n")
                        valid_count += 1
                except Exception as e:
                    print(f"Skipping invalid entry: {e}")
                    
        print(f"Data Generation Complete. Generated {valid_count} examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
