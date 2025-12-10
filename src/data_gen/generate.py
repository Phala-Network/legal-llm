import os
import json
import random
import glob
import re
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DataGenerator:
    def __init__(self, data_dir: str = "data", output_file: str = "training_data.jsonl", model: str = "openai/gpt-5.1"):
        self.data_dir = data_dir
        self.output_file = output_file
        self.client = OpenAI()
        self.model = model

    def _get_case_text(self, json_path: str) -> str:
        """Extracts text from the JSON case file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            opinions = data.get('casebody', {}).get('opinions', [])
            if not opinions:
                return ""
            
            # Combine opinions
            text = "\n\n".join([op.get('text', '') for op in opinions])
            return text
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            return ""

    def _generate_candidate(self, context_text: str) -> Optional[List[Dict]]:
        """Generates candidate training examples from the text."""
        prompt = f"""
        You are an expert legal data annotator.
        Task: Create 3 diverse "Agentic RAG" training examples based on the provided legal opinion.
        
        The goal is to teach an AI Agent how to research legal cases.
        The questions should require multi-step reasoning: understanding the facts -> formulating a search -> finding the law.
        
        Structure for each example:
        1. "question": A complex user question (e.g., "Why did the court reverse the decision regarding...").
        2. "thought": A chain-of-thought explaining the research strategy.
        3. "search_query": A specific, keyword-optimized search query to find this case or relevant precedence.
        4. "relevant_context": Verbatim excerpt from the text containing the answer.
        5. "answer": The final detailed answer derived from the context.

        Case Text (Truncated):
        {context_text[:8000]}

        Output format: JSON list of objects.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for legal data generation."},
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
            return None

    def _critic_review(self, example: Dict, context_text: str) -> bool:
        """
        Critic Model: Evaluates the quality of a generated example.
        Returns True if quality is high (>= 4/5), False otherwise.
        """
        critic_prompt = f"""
        Rate this training example on a scale of 1-5 based on:
        1. Faithfulness: Does the answer strictly follow the context?
        2. Relevance: Is the search query a good keyword representation of the question?
        3. Complexity: Is the question non-trivial (requires reasoning)?
        
        Example:
        Question: {example.get('question')}
        Thought: {example.get('thought')}
        Search: {example.get('search_query')}
        Answer: {example.get('answer')}
        Context Snippet: {example.get('relevant_context')[:200]}...

        Return JSON: {{"score": <int>, "reason": "<string>"}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": critic_prompt}],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            grading = json.loads(response.choices[0].message.content)
            score = grading.get("score", 0)
            if score >= 4:
                return True
            return False
        except Exception:
            return False

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
            text = self._get_case_text(json_file)
            if not text or len(text) < 500:
                continue

            candidates = self._generate_candidate(text)
            if not candidates:
                continue

            for cand in candidates:
                # Validate keys
                required_keys = ["question", "thought", "search_query", "relevant_context", "answer"]
                if not all(k in cand for k in required_keys):
                    continue

                # Critic Loop
                if self._critic_review(cand, text):
                    # Format for Agentic RAG Training
                    entry = {
                        "messages": [
                            {"role": "user", "content": cand["question"]},
                            {"role": "assistant", "content": f"<thought>{cand['thought']}</thought>\n<search>{cand['search_query']}</search>"},
                            {"role": "user", "content": f"Search Results:\n{cand['relevant_context']}\n"},
                            {"role": "assistant", "content": cand["answer"]}
                        ]
                    }
                    
                    # Write immediately (one-by-one)
                    with open(self.output_file, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(entry) + "\n")
                    
                    valid_count += 1
                    
        print(f"Data Generation Complete. Generated {valid_count} high-quality examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
