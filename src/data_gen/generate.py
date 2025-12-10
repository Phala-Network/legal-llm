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

    def _get_html_text(self, json_path: str) -> str:
        """Extracts text from the companion HTML file of a JSON metadata file."""
        html_path = json_path.replace('/json/', '/html/').replace('.json', '.html')
        if os.path.exists(html_path):
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple cleanup to remove tags but keep some structure if needed?
                    # For now, aggressive cleaning as before, but maybe we want to keep <p> later for chunking.
                    # The current requirement is just text for context.
                    clean = re.compile('<.*?>')
                    text = re.sub(clean, ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text
            except Exception as e:
                print(f"Error reading HTML {html_path}: {e}")
                return ""
        return ""

    def _generate_candidate(self, context_text: str) -> Optional[List[Dict]]:
        """Generates candidate training examples from the text."""
        prompt = f"""
        You are an expert legal data annotator.
        Task: Create 3 diverse "Agentic RAG" training examples based on the provided legal text.
        
        Structure for each example:
        1. "question": A user question (mix of specific fact retrieval and multi-hop reasoning).
        2. "thought": A chain-of-thought explaining what to search for.
        3. "search_query": The specific keyword query to execute.
        4. "relevant_context": Verbatim excerpt from the text containing the answer.
        5. "answer": The final answer derived from the context.

        Case Text (Truncated):
        {context_text[:6000]}

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
        3. Complexity: Is the question non-trivial?

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
            # print(f"Rejected (Score {score}): {grading.get('reason')}")
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
        with open(self.output_file, 'w') as out_f:
            for json_file in tqdm(selected_files, desc="Generating Data"):
                print(f"Processing {json_file}...")
                text = self._get_html_text(json_file)
                if not text or len(text) < 500:
                    print(f"Skipping {json_file}: text too short ({len(text)})")
                    continue

                print("Generating candidates...")
                candidates = self._generate_candidate(text)
                if not candidates:
                    print("No candidates generated.")
                    continue

                for cand in candidates:
                    print(" reviewing candidate...")
                    # Validate keys
                    required_keys = ["question", "thought", "search_query", "relevant_context", "answer"]
                    if not all(k in cand for k in required_keys):
                        continue

                    # Critic Loop
                    if self._critic_review(cand, text):
                        # Format for Agentic RAG Training (User -> Thought -> Search -> Context -> Answer)
                        # We represent the Search Tool Call convention.
                        # Note: This is a simplified "text-based" representation of tool calling.
                        entry = {
                            "messages": [
                                {"role": "user", "content": cand["question"]},
                                {"role": "assistant", "content": f"<thought>{cand['thought']}</thought>\n<search>{cand['search_query']}</search>"},
                                # The "tool output" is injected as a User message in many chat formats, or a specific Tool role.
                                # For simplicity here, we simulate the Search Result return.
                                {"role": "user", "content": f"Search Results:\n{cand['relevant_context']}\n"},
                                {"role": "assistant", "content": cand["answer"]}
                            ]
                        }
                        out_f.write(json.dumps(entry) + "\n")
                        out_f.flush()
                        valid_count += 1

        print(f"Data Generation Complete. Generated {valid_count} high-quality examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
