import os
import json
import random
import glob
import argparse
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import sys
import re
import time
import uuid

load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.retriever import CaseRetriever
from src.rag.case_parser import CaseParser


class BatchManager:
    def __init__(self, client: OpenAI):
        self.client = client

    def upload_file(self, file_path: str, purpose: str = "batch") -> str:
        print(f"Uploading {file_path} to OpenAI...")
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose=purpose)
        print(f"File uploaded. ID: {response.id}")
        return response.id

    def create_batch(self, input_file_id: str, description: str = "Batch Job") -> str:
        print(f"Creating Batch for File ID: {input_file_id}...")
        response = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        print(f"Batch created. ID: {response.id}")
        return response.id

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> str:
        print(
            f"Waiting for Batch {batch_id} to complete. Polling every {poll_interval}s..."
        )
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            print(f"Batch {batch_id} Status: {status}")

            if status == "completed":
                if batch.output_file_id:
                    return batch.output_file_id
                else:
                    # Completed but no output? Maybe empty
                    print("Batch completed but no output file ID found.")
                    return ""
            elif status in ["failed", "cancelled", "expired"]:
                print(f"Batch failed with status: {status}")
                if hasattr(batch, "errors") and batch.errors:
                    print(f"Errors: {batch.errors}")
                raise Exception(f"Batch process failed: {status}")

            time.sleep(poll_interval)

    def download_file(self, file_id: str, output_path: str):
        print(f"Downloading File {file_id} to {output_path}...")
        content = self.client.files.content(file_id).read()
        with open(output_path, "wb") as f:
            f.write(content)
        print("Download complete.")


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

        self.client = OpenAI()
        self.batch_manager = BatchManager(self.client)
        self.model = model or os.getenv("GENERATION_MODEL", "gpt-4o")
        self.parser = CaseParser(data_dir=data_dir)
        self.retriever = None

    def _init_retriever(self):
        if self.retriever:
            return
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
            text = self.parser.extract_full_text(data, include_recursive=True)
            return {
                "text": text,
                "id": str(data.get("id", "")),
                "name": data.get("name_abbreviation", data.get("name", "Unknown Case")),
                "file_path": json_path,
            }
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            return {}

    def _is_valid_case(self, json_path: str) -> bool:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = self.parser.extract_full_text(data, include_recursive=False)
            if len(text) < 500:
                return False
            return True
        except:
            return False

    def _get_full_recursive_text(self, metadata: Dict) -> str:
        file_path = metadata.get("file_path")
        if not file_path:
            return ""
        abs_path = (
            os.path.abspath(file_path)
            if os.path.exists(file_path)
            else os.path.join(os.getcwd(), file_path)
        )
        if not os.path.exists(abs_path):
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

    # -------------------------------------------------------------------------
    # AUTOMATED PIPELINE
    # -------------------------------------------------------------------------
    def run_pipeline(self, num_samples: int):
        pipeline_id = uuid.uuid4().hex[:8]
        print(f"Starting Automated Pipeline {pipeline_id}...")

        # Files
        f_queries_in = f"batch_{pipeline_id}_step1_queries.jsonl"
        f_queries_out = f"batch_{pipeline_id}_step1_output.jsonl"
        f_answers_in = f"batch_{pipeline_id}_step2_answers.jsonl"
        f_answers_out = f"batch_{pipeline_id}_step2_output.jsonl"

        map_1 = f"batch_{pipeline_id}_map1.json"
        map_2 = f"batch_{pipeline_id}_map2.json"

        # --- STAGE 1: Generate Queries ---
        print("\n=== STAGE 1: generating queries ===")
        # We need to temporarily re-point output_file
        original_output = self.output_file
        self.output_file = f_queries_in

        self.prepare_batch_queries(num_samples, map_1)

        if not os.path.exists(f_queries_in) or os.path.getsize(f_queries_in) == 0:
            print("No queries generated or empty file. Exiting.")
            return

        # Upload & Run Batch 1
        fid_1 = self.batch_manager.upload_file(f_queries_in)
        bid_1 = self.batch_manager.create_batch(
            fid_1, description=f"Pipeline {pipeline_id} Step 1 (Queries)"
        )
        out_fid_1 = self.batch_manager.wait_for_batch(bid_1)

        if not out_fid_1:
            print("Stage 1 failed to produce output.")
            return

        self.batch_manager.download_file(out_fid_1, f_queries_out)

        # --- STAGE 2: Process & Generate Answers ---
        print("\n=== STAGE 2: Running RAG & Preparing Answers ===")
        self.output_file = f_answers_in  # Set target for next stage generation
        self.process_and_prepare_answers(f_queries_out, map_1, map_2)

        if not os.path.exists(f_answers_in) or os.path.getsize(f_answers_in) == 0:
            print("No answers requests generated. Exiting.")
            return

        # Upload & Run Batch 2
        fid_2 = self.batch_manager.upload_file(f_answers_in)
        bid_2 = self.batch_manager.create_batch(
            fid_2, description=f"Pipeline {pipeline_id} Step 2 (Answers)"
        )
        out_fid_2 = self.batch_manager.wait_for_batch(bid_2)

        if not out_fid_2:
            print("Stage 2 failed to produce output.")
            return

        self.batch_manager.download_file(out_fid_2, f_answers_out)

        # --- STAGE 3: Finalize ---
        print("\n=== STAGE 3: Finalizing Dataset ===")
        self.output_file = original_output  # Restore final target
        self.finalize_dataset(f_answers_out, map_2)

        print(f"\nPipeline {pipeline_id} Complete! Final data in: {self.output_file}")

    # -------------------------------------------------------------------------
    # STAGE 1: Prepare Query Generation Batch
    # -------------------------------------------------------------------------
    def prepare_batch_queries(self, num_samples: int, map_file: str):
        """
        Scans files, selects candidates, and generates the JSONL for OpenAI Batch API (Stage 1).
        Map file saves: {custom_id: case_file_path}
        """
        json_files = glob.glob(
            os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True
        )
        # Filter processed
        json_files = [
            f for f in json_files if os.path.abspath(f) not in self.processed_files
        ]
        if not json_files:
            print("No new files to process.")
            return

        print(f"Found {len(json_files)} candidates. Filtering validity...")
        valid_files = []
        for f in tqdm(
            json_files[: num_samples * 2], desc="Filtering Files"
        ):  # Check slightly more than needed
            if len(valid_files) >= num_samples:
                break
            if self._is_valid_case(f):
                valid_files.append(f)

        # Shuffle/Select
        random.shuffle(valid_files)
        selected_files = valid_files[:num_samples]
        print(f"Selected {len(selected_files)} files for batch generation.")

        meta_map = {}

        with open(self.output_file, "w", encoding="utf-8") as f_out:
            for f_path in tqdm(selected_files, desc="Writing Batch Requests"):
                case_info = self._get_case_text(f_path)
                if not case_info:
                    continue

                custom_id = f"req-{uuid.uuid4()}"

                # Construct Prompt
                messages = self._construct_query_prompt(case_info)

                # Batch Request Body
                body = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "temperature": 0.7,
                    },
                }

                f_out.write(json.dumps(body) + "\n")
                meta_map[custom_id] = os.path.abspath(f_path)

        # Save Map
        with open(map_file, "w", encoding="utf-8") as f_map:
            json.dump(meta_map, f_map, indent=2)

        print(f"Stage 1 Prep Complete. Output: {self.output_file}")

    def _construct_query_prompt(self, case_info: Dict) -> List[Dict]:
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
        return [
            {"role": "system", "content": "Output valid JSON list only."},
            {"role": "user", "content": prompt},
        ]

    # -------------------------------------------------------------------------
    # STAGE 2: Process Queries -> Local RAG -> Prepare Answer Batch
    # -------------------------------------------------------------------------
    def process_and_prepare_answers(
        self, input_file: str, map_file: str, next_map_file: str
    ):
        """
        Reads Stage 1 Output (Batch results), runs RAG locally, and prepares Stage 2 Batch Request.
        """
        if not os.path.exists(map_file):
            print(f"Error: Map file {map_file} not found.")
            return

        with open(map_file, "r") as f:
            query_map = json.load(f)

        self._init_retriever()

        next_meta_map = {}

        with (
            open(input_file, "r", encoding="utf-8") as f_in,
            open(self.output_file, "w", encoding="utf-8") as f_out,
        ):

            for line in tqdm(f_in, desc="Processing Queries & Running RAG"):
                try:
                    res = json.loads(line)
                    custom_id = res.get("custom_id")

                    # Ensure successful response
                    if res.get("error"):
                        print(f"Skipping failed request {custom_id}: {res['error']}")
                        continue

                    response_body = res.get("response", {}).get("body", {})
                    content = (
                        response_body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if not content or not custom_id:
                        continue

                    case_path = query_map.get(custom_id)
                    if not case_path:
                        print(f"Warning: No map entry for {custom_id}")
                        continue

                    # Load original case text for the Direct Answer context
                    case_info = self._get_case_text(case_path)

                    # Parse Queries
                    queries = self._parse_queries_output(content)
                    if not queries:
                        continue

                    # Run RAG and prepare prompt items
                    items_to_process = []
                    for q_item in queries:
                        search_query = q_item.get("search_query")
                        retrieved_context_str = ""

                        if search_query and self.retriever:
                            retrieved_docs = self.retriever.retrieve(search_query, k=4)
                            for i, doc in enumerate(retrieved_docs):
                                # Format context for the model
                                name = doc.get("metadata", {}).get("name", "Unknown")
                                file_path = doc.get("metadata", {}).get("file_path", "")
                                full_text = self._get_full_recursive_text(
                                    doc.get("metadata", {})
                                )
                                text_snippet = (
                                    full_text if full_text else doc.get("text", "")
                                )[:3000]
                                retrieved_context_str += f"[Result {i+1}] {name} (File: {file_path})\n{text_snippet}...\n\n"

                        items_to_process.append(
                            {
                                "q_item": q_item,
                                "search_query": search_query,
                                "retrieved_context_str": retrieved_context_str,
                            }
                        )

                    # Create Batch Request for Answers (One request per case, containing all 10 items)
                    new_custom_id = f"ans-{uuid.uuid4()}"
                    messages = self._construct_answer_prompt(
                        items_to_process, case_info.get("text", "")
                    )

                    body = {
                        "custom_id": new_custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                            "temperature": 0.3,
                        },
                    }

                    f_out.write(json.dumps(body) + "\n")

                    next_meta_map[new_custom_id] = {
                        "original_file": case_path,
                        "items": items_to_process,
                    }

                except Exception as e:
                    print(f"Error processing line: {e}")

        # Save Next Map
        with open(next_map_file, "w", encoding="utf-8") as f_map:
            json.dump(next_meta_map, f_map, indent=2)

        print(f"Stage 2 Prep Complete. Output: {self.output_file}")

    def _parse_queries_output(self, content: str) -> List[Dict]:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if "queries" in data and isinstance(data["queries"], list):
                    return data["queries"]
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
            if isinstance(data, list):
                return data
            return []
        except:
            return []

    def _construct_answer_prompt(self, items: List[Dict], gold_text: str) -> List[Dict]:
        task_prompt = (
            "You are a legal AI assistant training to be strict and precise.\nTasks:\n"
        )

        for i, item in enumerate(items):
            q_text = item["q_item"]["question"]
            results = item["retrieved_context_str"]
            has_search = bool(item["search_query"])

            task_prompt += f"--- ITEM {i} ---\nQuestion: {q_text}\n"
            if has_search:
                task_prompt += f"Context (Search Results ONLY):\n{results if results else 'No results found.'}\n"
            else:
                task_prompt += f"Context (Gold Case):\n{gold_text[:10000]}\n"
            task_prompt += "\n"

        task_prompt += """
        IMPORTANT INSTRUCTIONS:
        1. **Strict RAG**: For Items with "Search Results ONLY", you MUST answer using ONLY that information. If the answer is not in the search results, state "I cannot answer based on the provided results."
        2. **CoT Enforcement**: You MUST use the `<thought>` block to generate a Chain of Thought.
           - Format: **IRAC** (Issue, Rule, Application, Conclusion).
        3. **Long Answer**: The content in `"answer"` must be a DETAILED paragraph.
        4. **Citations**: If using Search Results, cite them as [Case Name](<File Path>).
        
        Output a valid JSON list of objects:
        [
            { "item_id": 0, "thought": "...", "answer": "..." },
            ...
        ]
        """
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Output valid JSON object with an 'answers' key.",
            },
            {"role": "user", "content": task_prompt},
        ]

    # -------------------------------------------------------------------------
    # STAGE 3: Finalize Dataset
    # -------------------------------------------------------------------------
    def finalize_dataset(self, input_file: str, map_file: str):
        """
        Reads Stage 2 Output, combines with map data, and writes final JSONL training data.
        """
        if not os.path.exists(map_file):
            print(f"Error: Map file {map_file} not found.")
            return

        with open(map_file, "r") as f:
            answer_map = json.load(f)

        count = 0
        with (
            open(input_file, "r", encoding="utf-8") as f_in,
            open(self.output_file, "a", encoding="utf-8") as f_out,
            open(self.processed_log, "a") as f_log,
        ):

            for line in tqdm(f_in, desc="Finalizing Dataset"):
                try:
                    res = json.loads(line)
                    custom_id = res.get("custom_id")

                    if res.get("error"):
                        print(f"Skipping failed answer batch {custom_id}")
                        continue

                    response_body = res.get("response", {}).get("body", {})
                    content = (
                        response_body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )

                    if not content or not custom_id:
                        continue

                    meta = answer_map.get(custom_id)
                    if not meta:
                        continue

                    items = meta["items"]
                    answers = self._parse_answers_output(content)

                    for ans in answers:
                        item_id = ans.get("item_id")
                        if item_id is None or not isinstance(item_id, int):
                            continue
                        if item_id < 0 or item_id >= len(items):
                            continue

                        item = items[item_id]

                        messages = [
                            {"role": "user", "content": item["q_item"]["question"]}
                        ]

                        if item["search_query"]:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": f"<thought>{ans.get('thought', '')}</thought>\n<search>{item['search_query']}</search>",
                                }
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": f"Search Results:\n{item['retrieved_context_str']}\n",
                                }
                            )

                        messages.append(
                            {
                                "role": "assistant",
                                "content": ans.get("answer", "I cannot answer."),
                            }
                        )

                        entry = {"messages": messages}
                        f_out.write(json.dumps(entry) + "\n")
                        count += 1

                    original_file = meta.get("original_file")
                    if original_file and original_file not in self.processed_files:
                        f_log.write(os.path.abspath(original_file) + "\n")
                        self.processed_files.add(os.path.abspath(original_file))

                except Exception as e:
                    print(f"Error finalizing line: {e}")

        print(f"Finalization Complete. Added {count} examples to {self.output_file}")

    def _parse_answers_output(self, content: str) -> List[Dict]:
        try:
            data = None
            try:
                data = json.loads(content)
            except:
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

            if not data:
                return []

            if isinstance(data, dict):
                if "answers" in data and isinstance(data["answers"], list):
                    return data["answers"]
                for k, v in data.items():
                    if isinstance(v, list):
                        return v
                if "answer" in data:
                    return [data]

            if isinstance(data, list):
                return data
            return []
        except:
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add pipeline option
    parser.add_argument(
        "--pipeline", action="store_true", help="Run full automated pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["queries", "answers", "finalize"],
        help="Manual processing stage",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of files to process (Stage 1 or Pipeline)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file (Output from OpenAI Batch for Stage 2/3)",
    )
    parser.add_argument(
        "--output_file", type=str, default="training_data.jsonl", help="Output file"
    )
    parser.add_argument(
        "--map_file", type=str, default="batch_map.json", help="Metadata map file"
    )
    parser.add_argument(
        "--next_map_file",
        type=str,
        default="batch_map_next.json",
        help="Next stage metadata map file",
    )

    args = parser.parse_args()

    gen = DataGenerator(output_file=args.output_file)

    if args.pipeline:
        gen.run_pipeline(num_samples=args.num_samples)
    elif args.stage == "queries":
        gen.prepare_batch_queries(num_samples=args.num_samples, map_file=args.map_file)
    elif args.stage == "answers":
        if not args.input_file:
            print("Error: --input_file is required for 'answers' stage")
        else:
            gen.process_and_prepare_answers(
                input_file=args.input_file,
                map_file=args.map_file,
                next_map_file=args.next_map_file,
            )
    elif args.stage == "finalize":
        if not args.input_file:
            print("Error: --input_file is required for 'finalize' stage")
        else:
            gen.finalize_dataset(input_file=args.input_file, map_file=args.map_file)
    else:
        print("Please specify --pipeline or --stage.")
