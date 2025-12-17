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
import re
import time

load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.retriever import CaseRetriever
from src.rag.case_parser import CaseParser


class DataGenerator:
    def __init__(
        self,
        data_dir: str = "data",
        output_file: str = "training_data.jsonl",
        model: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.output_file = output_file
        self.processed_log = "processed_files.txt"
        self.processed_files = set()
        if os.path.exists(self.processed_log):
            with open(self.processed_log, "r") as f:
                self.processed_files = set(f.read().splitlines())
        print(f"Loaded {len(self.processed_files)} processed files from history.")
        self.client = OpenAI()
        self.model = model or os.getenv("GENERATION_MODEL", "openai/gpt-4o")

        # Initialize Parser
        self.parser = CaseParser(data_dir=data_dir)

        print("Initializing Real RAG Retriever...")
        import chromadb

        try:
            temp_client = chromadb.PersistentClient(path="chroma_db")
            cols = [
                c.name
                for c in temp_client.list_collections()
                if c.name.startswith("law_cases")
            ]
            target_col = cols[0] if cols else "law_cases"
            self.retriever = CaseRetriever(collection_name=target_col)
        except Exception as e:
            print(
                f"Warning: Could not initialize retriever ({e}). retrieval capabilities will fail."
            )
            self.retriever = None

    def _get_case_text(self, json_path: str) -> Dict:
        """Extracts text and metadata from the JSON case file using CaseParser (Recursive)."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Use Parser for Recursive Text
            text = self.parser.extract_full_text(data, include_recursive=True)

            return {
                "text": text,
                "id": str(data.get("id", "")),
                "name": data.get("name_abbreviation", data.get("name", "Unknown Case")),
            }
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            return {}

    def _is_valid_case(self, json_path: str) -> bool:
        """
        Pre-checks if a case file has sufficient content to be useful.
        This avoids processing empty cases like '197/json/0129-01.json'.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Quick check on text length without full recursive overhead if possible,
            # but we need the text to be sure.
            # Using parser.extract_full_text is safe as it's local.
            text = self.parser.extract_full_text(
                data, include_recursive=False
            )  # Check base text first
            if len(text) < 500:
                # If base text is short, maybe recursive text helps?
                # But usually if main opinion is empty, it's a bad case for "Gold Case" generation.
                return False
            return True
        except:
            return False

    def _get_full_recursive_text(self, metadata: Dict) -> str:
        """
        Attempts to load the full recursive text for a retrieved document
        using the 'file_path' in metadata.
        """
        file_path = metadata.get("file_path")
        if not file_path:
            return ""  # Fallback to snippet if no path

        # Resolve path
        # metadata['file_path'] is likely relative to project root or data dir
        # e.g., "data/cal-rptr-3d/..."
        # We need to ensure it's absolute
        abs_path = (
            os.path.abspath(file_path)
            if os.path.exists(file_path)
            else os.path.join(os.getcwd(), file_path)
        )

        if not os.path.exists(abs_path):
            # Try relative to self.data_dir
            abs_path = os.path.join(self.data_dir, file_path)

        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self.parser.extract_full_text(data, include_recursive=True)
            except Exception as e:
                print(f"Error loading recursive text from {abs_path}: {e}")
                return ""
        return ""

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
        {{
            "queries": [
                {{
                    "type": "complex", # or "simple" or "negative"
                    "question": "...",
                    "search_query": "..." # or null
                }},
                ...
            ]
        }}

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
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            print(f"DEBUG: Raw Query Response: {data}")

            if isinstance(data, dict):
                # Check for explicit keys
                if "queries" in data and isinstance(data["queries"], list):
                    return data["queries"]

                # Check for dynamic keys (like "items": [...])
                for k, v in data.items():
                    if isinstance(v, list):
                        return v

                # Fallback: if data looks like a single query object
                if "question" in data and "type" in data:
                    return [data]

            # If it's a list (shouldn't happen with response_format json_object but for safety)
            return data if isinstance(data, list) else []

        except Exception as e:
            print(f"Query Gen error: {e}")
            return []

    def _generate_answers_batch(self, items: List[Dict], gold_text: str) -> List[Dict]:
        """
        Step 3 (Batched): Generate final answers for multiple questions in one go.
        """
        # Construct a combined prompt
        # Prepare prompt
        task_prompt = (
            "You are a legal AI assistant training to be strict and precise.\n"
        )

        # We need to construct item-specific contexts.
        # But this is a BATCH generation. We can't have different system instructions effectively if we share one prompt.
        # Actually we can: "For Item 0, use this context..."

        task_prompt += "Tasks:\n"

        for i, item in enumerate(items):
            q_text = item["q_item"]["question"]
            results = item.get("retrieved_context_str", "")
            has_search = bool(item.get("search_query"))

            task_prompt += f"--- ITEM {i} ---\nQuestion: {q_text}\n"

            if has_search:
                # STRICT RAG: Hide Gold Case. Show ONLY search results.
                task_prompt += f"Context (Search Results ONLY):\n{results if results else 'No results found.'}\n"
            else:
                # Direct Answer: Use Gold Case
                task_prompt += (
                    f"Context (Gold Case):\n{gold_text[:10000]}\n"  # Truncated Gold
                )

            task_prompt += "\n"

        task_prompt += """
        IMPORTANT INSTRUCTIONS:
        1. **Strict RAG**: For Items with "Search Results ONLY", you MUST answer using ONLY that information. If the answer is not in the search results, state "I cannot answer based on the provided results."
        2. **CoT Enforcement**: You MUST use the `<thought>` block to generate a Chain of Thought.
           - Format: **IRAC** (Issue, Rule, Application, Conclusion).
           - Do not be shallow. Explain your reasoning steps.
        3. **Long Answer**: The content in `"answer"` must be a DETAILED, STANDALONE paragraph (approx. 3-5 sentences). Avoid short "Yes/No" or one-line answers.
        4. **Citations**: If using Search Results, cite them as [Case Name](<File Path>). The File Path is provided in the search results headers.
        
        Output a valid JSON list of objects:
        [
            { "item_id": 0, "thought": "Issue: ... Rule: ... Application: ... Conclusion: ...", "answer": "The court held that..." },
            ...
        ]
        """

        def parse_json_robust(text):
            # Try standard load
            try:
                return json.loads(text)
            except:
                pass
            # Try to find list brackets
            try:
                match = re.search(r"\[.*\]", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except:
                pass
            return None

        # Retry Loop
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Output valid JSON object with an 'answers' key containing the list.",
                        },
                        {"role": "user", "content": task_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                content = response.choices[0].message.content

                data = parse_json_robust(content)
                if not data:
                    continue

                # Normalize keys
                if isinstance(data, dict):
                    # Check for explicit 'answers' key
                    if "answers" in data and isinstance(data["answers"], list):
                        return data["answers"]

                    # Check for any list value
                    for k, v in data.items():
                        if isinstance(v, list):
                            return v

                    # Fallback: if data is a single object (e.g. from fallback loop), wrap it
                    if "answer" in data or "thought" in data:
                        return [data]

                if isinstance(data, list):
                    return data
            except Exception as e:
                print(f"Batch Gen Error (Attempt {attempt+1}): {e}")
                time.sleep(1)

        return []

    def run(self, num_samples: int = 10):
        json_files = glob.glob(
            os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True
        )
        if not json_files:
            print("No data files found.")
            return

        # Pre-filter files to remove empty ones
        print("Pre-examining files for validity...")
        valid_files = []
        for f in tqdm(json_files, desc="Filtering Files"):
            if self._is_valid_case(f):
                valid_files.append(f)

        print(f"Found {len(valid_files)} valid cases out of {len(json_files)}.")
        json_files = valid_files  # Replace with valid list

        # Filter out already processed files
        json_files = [
            f for f in json_files if os.path.abspath(f) not in self.processed_files
        ]
        print(f"Remaining candidates after filtering history: {len(json_files)}")

        # Shuffle files to ensure randomness
        random.shuffle(json_files)

        valid_count = 0

        # We want to generate 'num_samples' VALID examples.
        # So we iterate through the shuffled list until we hit the count or run out.
        pbar = tqdm(total=num_samples, desc="Generating Data")

        file_idx = 0
        while valid_count < num_samples and file_idx < len(json_files):
            json_file = json_files[file_idx]
            file_idx += 1

            # print(f"DEBUG: Checking {json_file}")
            case_info = self._get_case_text(json_file)

            # Step 1: Generate Queries
            query_batch = self._generate_queries(case_info)
            if not query_batch:
                print(f"DEBUG: No valid queries generated for {json_file}")
                continue

            print(f"DEBUG: Generated {len(query_batch)} queries for {json_file}")

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
                        real_id = (
                            doc["id"].split("_")[0] if "_" in doc["id"] else doc["id"]
                        )
                        name = doc.get("metadata", {}).get("name", "Unknown Case")

                        # Recursive Retrieval Step
                        metadata = doc.get("metadata", {})
                        file_path = metadata.get("file_path", "unknown_path")

                        full_recursive_text = self._get_full_recursive_text(metadata)

                        # Use recursive text if available, else fallback to snippet
                        # Truncate strictly to avoid context window explosion (e.g. 3000 chars per result)
                        content_text = (
                            full_recursive_text if full_recursive_text else doc["text"]
                        )
                        content_text = content_text[:3000]

                        context_str_final += f"[Result {i+1}] {name} (File: {file_path})\n{content_text}...\n\n"

                items_to_process.append(
                    {
                        "q_item": q_item,
                        "search_query": search_query,
                        "retrieved_context_str": context_str_final,
                    }
                )

            print(
                f"DEBUG: Generated {len(items_to_process)} items to process for {json_file}"
            )

            # Step 3: Batched Answer Generation
            if not items_to_process:
                continue

            answers = self._generate_answers_batch(items_to_process, case_info["text"])
            if not answers:
                continue

            # Match answers back
            if len(answers) != len(items_to_process):
                print(
                    f"Warning: Batch size mismatch (Sent {len(items_to_process)}, Got {len(answers)}). Matching what we can."
                )

            # Process answers using item_id
            for ans_data in answers:
                try:
                    # Robustly get item_id
                    item_id = ans_data.get("item_id")
                    if item_id is None:
                        # If single item batch or just one answer, maybe we can assume?
                        # But safe is to skip or try heuristic.
                        # For now, let's log and skip if strict.
                        print("Warning: Answer missing item_id, skipping.")
                        continue

                    if (
                        not isinstance(item_id, int)
                        or item_id < 0
                        or item_id >= len(items_to_process)
                    ):
                        print(f"Warning: Invalid item_id {item_id}, skipping.")
                        continue

                    item = items_to_process[item_id]
                    q_item = item["q_item"]
                    search_query = item["search_query"]
                    context_str_final = item["retrieved_context_str"]

                    # Construct Final Entry
                    messages = [{"role": "user", "content": q_item["question"]}]

                    if search_query:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"<thought>{ans_data.get('thought', 'Thinking...')}</thought>\n<search>{search_query}</search>",
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Search Results:\n{context_str_final}\n",
                            }
                        )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": ans_data.get("answer", "I cannot answer."),
                        }
                    )

                    entry = {"messages": messages}

                    with open(self.output_file, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps(entry) + "\n")
                    valid_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Skipping entry in batch: {e}")

            # Mark file as processed after using it
            with open(self.processed_log, "a") as f:
                f.write(os.path.abspath(json_file) + "\n")
            self.processed_files.add(os.path.abspath(json_file))

        pbar.close()
        print(f"Data Generation Complete. Generated {valid_count} examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
