import os
import json
import random
import glob
import time
import re
import uuid
from typing import List, Dict, Optional
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import sys

load_dotenv()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rag.retriever import CaseRetriever
from src.rag.case_parser import CaseParser


class BaseGenerator:
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
        self.model = model or os.getenv("GENERATION_MODEL", "gpt-4o")
        self.parser = CaseParser(data_dir=data_dir)
        self.retriever = None

    def _init_retriever(self):
        if self.retriever:
            return
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
            print(f"Warning: Could not initialize retriever ({e}).")
            self.retriever = None

    def _get_case_text(self, json_path: str) -> Dict:
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
            return len(text) >= 500
        except:
            return False

    def _get_full_recursive_text(self, metadata: Dict) -> str:
        file_path = metadata.get("file_path")
        if not file_path:
            return ""
        abs_path = (
            os.path.abspath(file_path)
            if os.path.exists(file_path)
            else os.path.join(self.data_dir, file_path)
        )
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self.parser.extract_full_text(data, include_recursive=True)
            except:
                pass
        return ""

    def get_valid_case_files(self, num_samples: int) -> List[str]:
        """
        Finds json files by sampling directories first to avoid scanning the entire dataset.
        Returns a list of up to num_samples paths.
        """
        # Get top-level directories in data_dir (e.g. jurisdiction folders)
        try:
            top_dirs = [
                d
                for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
                and not d.startswith(".")
            ]
        except OSError:
            print(f"Error accessing data directory: {self.data_dir}")
            return []

        if not top_dirs:
            print("No directories found in data directory.")
            return []

        random.shuffle(top_dirs)

        valid_files = []
        # We need to find `num_samples` valid files.
        # We'll iterate through shuffled top_dirs, and for each, look for json files.
        # This is much faster than running glob on everything.

        print(f"Sampling directories to find {num_samples} valid files...")
        pbar = tqdm(total=num_samples)

        for d in top_dirs:
            if len(valid_files) >= num_samples:
                break

            # Look for json files in this directory (recursively within this jurisdiction)
            dir_path = os.path.join(self.data_dir, d)
            # Limit glob to avoid hanging on massive dirs if they exist
            # But usually jurisdiction dirs are manageable or subdivided.
            # Let's glob recursively inside this ONE dir.
            candidates = glob.glob(
                os.path.join(dir_path, "**", "*.json"), recursive=True
            )
            random.shuffle(candidates)

            for f in candidates:
                if len(valid_files) >= num_samples:
                    break

                abs_f = os.path.abspath(f)
                if abs_f in self.processed_files:
                    continue

                if self._is_valid_case(f):
                    valid_files.append(f)
                    pbar.update(1)

        pbar.close()
        return valid_files

    def augment_queries_with_context(self, queries: List[Dict]) -> List[Dict]:
        """
        Takes a list of query items (from LLM output), performs retrieval if search_query is present,
        and returns a list of items ready for answer generation.
        """
        items_to_process = []
        for q_item in queries:
            search_query = q_item.get("search_query")
            context_str = ""
            if search_query and self.retriever:
                retrieved = self.retriever.retrieve(search_query, k=4)
                for i, doc in enumerate(retrieved):
                    cid = doc["id"].split("_")[0]
                    name = doc.get("metadata", {}).get("name", "Unknown")
                    text = (
                        self._get_full_recursive_text(doc.get("metadata", {}))
                        or doc["text"]
                    )
                    context_str += (
                        f"[Result {i+1}] {name} (ID: {cid})\n{text[:3000]}...\n\n"
                    )

            items_to_process.append(
                {
                    "q_item": q_item,
                    "search_query": search_query,
                    "retrieved_context_str": context_str,
                }
            )
        return items_to_process

    def _construct_query_prompt(self, case_info: Dict) -> List[Dict]:
        context_text = case_info.get("text", "")
        case_id = case_info.get("id", "unknown")
        case_name = case_info.get("name", "Unknown Case")

        distribution = self._get_random_query_distribution()
        persona = random.choice(self.PERSONAS)

        prompt = f"""
        {persona}
        Task: Create a diverse set of training queries based on the provided legal text.

        Target Distribution (Approximate):
        1. [COMPLEX] ({distribution['complex']}): Multi-step reasoning where {case_name} (ID: {case_id}) is the primary answer source. `search_query` must be populated.
        2. [SIMPLE] ({distribution['simple']}): Specific fact retrieval from this case. `search_query` can be populated or empty.
        3. [GENERAL] ({distribution['general']}): General legal concept questions (e.g., "what is personal injury law?") that DON'T need search. `search_query` must be empty.
        4. [NEGATIVE] ({distribution['negative']}): Questions that are barely related or out-of-scope. `search_query` must be empty.

        Output Structure (JSON ONLY):
        {{
            "queries": [
                {{ "type": "complex", "question": "...", "search_query": "..." }},
                {{ "type": "general", "question": "...", "search_query": null }},
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

    PERSONAS = [
        "You are an expert legal data annotator focused on creating challenging, realistic bar-exam style questions.",
        "You are a curious law student acting as a user who needs specific details from cases.",
        "You are a legal researcher constructing a dataset for training a high-precision legal AI.",
        "You are a skeptic trying to find edge cases where the model might hallucinate or fail.",
    ]

    INSTRUCTIONS_TEMPLATES = [
        {
            "name": "IRAC_Strict",
            "prompt": "Use the traditional IRAC (Issue, Rule, Application, Conclusion) format for the 'thought' field. Be very structured.",
        },
        {
            "name": "Free_CoT",
            "prompt": "Think through the problem step-by-step in the 'thought' field. Explore multiple interpretations before settling on an answer. Do not use a fixed header structure.",
        },
        {
            "name": "Socratic_Review",
            "prompt": "In your 'thought' process, first ask yourself what information is missing, then evaluate the provided text, and finally construct the answer. Be self-critical.",
        },
        {
            "name": "Debate",
            "prompt": "Draft arguments for and against the conclusion in the 'thought' field. Weigh the evidence from the text before providing the final answer.",
        },
    ]

    def _get_random_query_distribution(self):
        # Randomize the mix to avoid fixed patterns
        total = 10
        complex_q = random.randint(3, 6)
        remaining = total - complex_q
        simple_q = random.randint(1, remaining - 1) if remaining > 1 else 1
        remaining = remaining - simple_q
        general_q = random.randint(0, remaining)
        negative_q = remaining - general_q
        return {
            "complex": complex_q,
            "simple": simple_q,
            "general": general_q,
            "negative": negative_q,
        }

    def _construct_answer_prompt(self, items: List[Dict], gold_text: str) -> List[Dict]:
        task_prompt = (
            "You are a legal AI assistant training to be strict and precise.\nTasks:\n"
        )
        for i, item in enumerate(items):
            q_text = item["q_item"]["question"]
            results = item["retrieved_context_str"]
            has_search = bool(item["search_query"])

            task_prompt += f"--- ITEM {i} ---\nQuestion: {q_text}\n"
            if item["q_item"].get("type") == "general":
                task_prompt += "Context (General Knowledge): You may use your internal knowledge for this general question.\n"
            elif has_search:
                task_prompt += f"Context (Search Results ONLY):\n{results if results else 'No results found.'}\n"
            else:
                task_prompt += f"Context (Gold Case):\n{gold_text[:10000]}\n"
            task_prompt += "\n"

        instruction_style = random.choice(self.INSTRUCTIONS_TEMPLATES)

        task_prompt += f"""
        IMPORTANT INSTRUCTIONS:
        1. **Strict RAG**: If "Search Results ONLY" are provided, use ONLY that info. If not found, tell the user you couldn't find it.
        2. **CoT Style**: {instruction_style['prompt']}
        3. **Thought Content**: The 'thought' field must contain your raw internal reasoning trace. You can use <thought> tags internally if helpful, but the entire field will be wrapped in tags later.
        4. **Citations**: Cite as [Case Name](ID: <case_id>).
        5. **Complete Output**: Ensure the "answers" list contains exactly {len(items)} items.

        Output valid JSON:
        {{
            "answers": [
                {{ "item_id": 0, "thought": "...", "answer": "..." }},
                {{ "item_id": 1, "thought": "...", "answer": "..." }},
                ...
                {{ "item_id": {len(items)-1}, "thought": "...", "answer": "..." }}
            ]
        }}
        """
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Output valid JSON object with an 'answers' key.",
            },
            {"role": "user", "content": task_prompt},
        ]

    def _parse_json_robust(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return None

    def _parse_queries_output(self, content: str) -> List[Dict]:
        data = self._parse_json_robust(content)
        if isinstance(data, dict):
            if "queries" in data:
                return data["queries"]
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data if isinstance(data, list) else []

    def _parse_answers_output(self, content: str) -> List[Dict]:
        data = self._parse_json_robust(content)
        if isinstance(data, dict):
            if "answers" in data:
                return data["answers"]
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data if isinstance(data, list) else []

    def construct_final_messages(self, item: Dict, ans_data: Dict) -> List[Dict]:
        """
        Constructs the final list of messages for fine-tuning.
        item: dict containing 'q_item' (question), 'search_query', 'retrieved_context_str'
        ans_data: dict containing 'thought' and 'answer'
        """
        msgs = [{"role": "user", "content": item["q_item"]["question"]}]

        thought_content = ans_data.get("thought", "").strip()
        # Ensure thought is wrapped if not empty
        if thought_content and not thought_content.startswith("<thought>"):
            thought_content = f"<thought>{thought_content}</thought>"
        elif not thought_content:
            thought_content = "<thought>Thinking Process...</thought>"

        if item["search_query"]:
            # Multi-step with search
            assistant_msg = (
                f"{thought_content}\n<search>{item['search_query']}</search>"
            )
            msgs.append({"role": "assistant", "content": assistant_msg})
            msgs.append(
                {
                    "role": "user",
                    "content": f"Search Results:\n{item['retrieved_context_str']}",
                }
            )
            msgs.append({"role": "assistant", "content": ans_data.get("answer")})
        else:
            # Direct answer with thought
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"{thought_content}\n{ans_data.get('answer')}",
                }
            )
        return msgs
