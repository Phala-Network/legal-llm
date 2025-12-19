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

    def _construct_query_prompt(self, case_info: Dict) -> List[Dict]:
        context_text = case_info.get("text", "")
        case_id = case_info.get("id", "unknown")
        case_name = case_info.get("name", "Unknown Case")

        prompt = f"""
        You are an expert legal data annotator.
        Task: Create diverse training queries strategy based on the provided legal opinion.

        Distribution:
        1. [COMPLEX] (4): Multi-step reasoning where {case_name} (ID: {case_id}) is the primary answer source. `search_query` should be populated.
        2. [SIMPLE] (4): Specific fact retrieval from this case. `search_query` can be populated or empty.
        3. [GENERAL] (2): General legal concept questions (e.g., "what is personal injury law?") that DON'T need search. `search_query` should be empty.
        4. [NEGATIVE] (1): Questions that are barely related or out-of-scope. `search_query` should be empty.

        Output Structure:
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

        task_prompt += f"""
        IMPORTANT INSTRUCTIONS:
        1. **Strict RAG**: If "Search Results ONLY" are provided, use ONLY that info. If not found, tell the user you couldn't find it in the search results.
        2. **CoT**: Use IRAC (Issue, Rule, Application, Conclusion) format for the 'thought' field. Do not include <thought> tags in the final 'answer'.
        3. **Citations**: Cite as [Case Name](ID: <case_id>).
        4. **Complete Output**: Ensure the "answers" list contains exactly {len(items)} items, matching the IDs provided (0 to {len(items)-1}).

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
