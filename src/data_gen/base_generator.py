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

    def _log_prompt(self, stage: str, prompt: str, response: str):
        """
        Logs the prompt and response to a debug file for inspection.
        """
        log_file = "generation_debug.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{separator}\n")
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"STAGE: {stage}\n")
                f.write(f"{separator}\n")
                f.write(f"PROMPT:\n{prompt}\n")
                f.write(f"{separator}\n")
                f.write(f"RESPONSE:\n{response}\n")
                f.write(f"{separator}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

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

        # Select a random style
        selected_style = random.choice(self.QUERY_STYLES)

        # Adjust distribution based on complexity bias if needed,
        # but for now we'll stick to the base distribution with potential minor tweaks if we wanted.
        distribution = self._get_random_query_distribution()

        selected_cot = random.choice(self.COT_STRATEGIES)

        prompt = f"""
        Your persona: {selected_style['style_name']}
        {selected_style['instruction']}

        Task: Create a diverse set of training queries based on the provided legal text.

        Reasoning Strategy: {selected_cot['prompt']}

        Target Distribution for this batch:
        1. [COMPLEX] * {distribution['complex']}: Multi-step reasoning. Primary focus for '{selected_style['style_name']}' style.
        2. [SIMPLE] * {distribution['simple']}: Specific fact retrieval.
        3. [GENERAL] * {distribution['general']}: General legal concepts.
        4. [NEGATIVE] * {distribution['negative']}: Out-of-scope.

        Output Structure (JSON ONLY):
        {{
            "queries": [
                {{
                    "type": "complex",
                    "thought": "Step-by-step reasoning using {selected_cot['name']} strategy on what to ask...",
                    "question": "...",
                    "search_query": "..."
                }},
                {{
                    "type": "general",
                    "thought": "Reasoning...",
                    "question": "...",
                    "search_query": null,
                    "answer": "Final answer for general/simple questions that don't need search."
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
        self._log_prompt(
            f"Pass 1 (Query Generation - {selected_cot['name']})",
            prompt,
            "(N/A - Prompt Construction)",
        )
        return [
            {"role": "system", "content": "Output valid JSON list only."},
            {"role": "user", "content": prompt},
        ], selected_cot["name"]

    QUERY_STYLES = [
        {
            "style_name": "The Senior Partner (Concise/Demanding)",
            "instruction": "Generate questions that are brief, direct, and demand high-level analysis. The user sounds impatient. The search query must be highly technical using boolean operators if possible.",
            "complexity_bias": "complex",
        },
        {
            "style_name": "The Pro Se Litigant (Verbose/Confused)",
            "instruction": "Generate questions that are overly wordy, emotional, and mix irrelevant details with the legal issue. The search query needs to extract the core legal keywords from the noise.",
            "complexity_bias": "simple",
        },
        {
            "style_name": "The Law Clerk (Procedural/Specific)",
            "instruction": "Focus questions on procedural history, jurisdiction, and specific motions. The user is detail-oriented. The search query should target specific procedural keywords.",
            "complexity_bias": "complex",
        },
        {
            "style_name": "The Layman (Generalist)",
            "instruction": "Generate questions using plain English, avoiding legal jargon. The user asks 'Can I sue?' rather than 'Is there a cause of action?'. The search query must bridge the gap between lay terms and legal terms.",
            "complexity_bias": "general",
        },
    ]

    OUTPUT_FORMATS = [
        "Plain Text Paragraphs (Standard)",
        "Bulleted List (Quick Summary)",
        "Formal Legal Memo (Header: To, From, Re, Date)",
        "Client Email (Professional but accessible)",
        "Judicial Opinion Style (Formal, authoritative)",
    ]

    CONSTRAINTS = [
        "Be extremely verbose. Explain every legal term used.",
        "Be ruthless with conciseness. Use fewer than 100 words for the final answer.",
        "Constraint: Cite the specific page number or paragraph if available in the text.",
        "Constraint: Explain it as if the user is a 1st-year law student (didactic tone).",
        "Constraint: Structure the answer with 'Key Holding', 'Reasoning', and 'Dicta' sections.",
    ]

    COT_STRATEGIES = [
        {
            "name": "Sequential_Logic",
            "prompt": "Strategy: Sequential Analysis. In the 'thought' field, map the logic linearly: 'First, I will identify the jurisdiction. Second, I will check the standing. Third, I will apply the rule to the facts.' Do not skip steps.",
        },
        {
            "name": "Adversarial_Critical",
            "prompt": "Strategy: The Devil's Advocate. In the 'thought' field, assume the initial intuition is wrong. Look for contradictions, overruled precedents, or distinguishing facts in the search results. Use phrases like 'However, looking closer at...' or 'A potential counter-argument is...'",
        },
        {
            "name": "Persona_Academic",
            "prompt": "Strategy: The Legal Scholar. In the 'thought' field, focus on the 'why' and the policy behind the rule. Connect the specific facts to broader legal principles. Discuss the intent of the court.",
        },
        {
            "name": "Persona_Data_Analyst",
            "prompt": "Strategy: Evidence Extraction. In the 'thought' field, treat the text as a dataset. extract dates, names, amounts, and citations explicitly before synthesizing the answer. If data is missing, flag it immediately.",
        },
        {
            "name": "IRAC_Strict",
            "prompt": "Strategy: IRAC Format. In the 'thought' field, explicitly label sections: [ISSUE], [RULE], [ANALYSIS], [CONCLUSION]. Ensure the analysis section connects specific facts to the rule.",
        },
    ]

    def _get_random_query_distribution(self):
        # Randomize the mix to avoid fixed patterns, but ensure negative_q is always 1
        total = 10
        negative_q = 1

        # Distributed remaining 9 among complex, simple, general
        remaining = total - negative_q

        complex_q = random.randint(3, 6)
        remaining -= complex_q

        # Ensure at least 1 simple, and try to leave room for general
        if remaining > 1:
            simple_q = random.randint(1, remaining - 1)
        else:
            simple_q = remaining

        general_q = remaining - simple_q

        return {
            "complex": complex_q,
            "simple": simple_q,
            "general": general_q,
            "negative": negative_q,
        }

    def construct_answer_conversations(
        self, items: List[Dict], gold_text: str
    ) -> List[List[Dict]]:
        """
        Generates a list of conversation histories (message lists), one for each item.
        Each conversation includes the Pass 1 thought and setup for the model to continue reasoning.
        """
        conversations = []

        for i, item in enumerate(items):
            q_item = item["q_item"]
            if q_item.get("answer") and not q_item.get("search_query"):
                # Already answered in Pass 1
                continue

            q_text = q_item["question"]
            results = item.get("retrieved_context_str", "")
            pass1_thought = q_item.get("thought", "Thinking...")

            # Retrieve the Strategy used in Pass 1 (injected into q_item during processing)
            strategy_name = q_item.get("cot_strategy_name", "Sequential_Logic")
            # Find the prompt text
            cot_prompt = next(
                (
                    s["prompt"]
                    for s in self.COT_STRATEGIES
                    if s["name"] == strategy_name
                ),
                "Think step by step.",
            )

            fmt = random.choice(self.OUTPUT_FORMATS)
            constraint = random.choice(self.CONSTRAINTS)

            system_msg = f"""You are a legal AI assistant.

            INSTRUCTIONS:
            1. **Reasoning Strategy**: Continue the reasoning process using this strategy: {cot_prompt}
            2. **Output Format**: {fmt}
            3. **Constraint/Tone**: {constraint}
            4. **Strict RAG**: Use the provided Search Results. If answer not found, state explicitly. Do not use outside knowledge unless it's a general question.
            5. **Citations**: only cite the case provided in the context [Case Name](ID: <case_id>).

            Output valid JSON: {{ "thought": "...", "answer": "..." }}
            """

            # Construct Conversation History
            # User: Question
            # Assistant: <thought>Pass 1 thought...</thought> <search>query</search>
            # User (Tool): Results

            # Note: We want to "force" the model to see its previous thought.
            # We can format this as a chat history.

            msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q_text},
            ]

            assistant_content = f"<thought>{pass1_thought}</thought>"
            if q_item.get("search_query"):
                assistant_content += f"\n<search>{q_item['search_query']}</search>"

            msgs.append({"role": "assistant", "content": assistant_content})

            # The "Tool Output" or "Search Results" coming from the user side (or tool role)
            msgs.append(
                {
                    "role": "user",
                    "content": f"Search Results:\n{results if results else 'No results found.'}\n\nPlease provide the final answer based on the above.",
                }
            )

            conversations.append(msgs)

            # Log for debug
            self._log_prompt(
                f"Pass 2 (Item {i} - {strategy_name})",
                json.dumps(msgs, indent=2),
                "(N/A - Prompt Construction)",
            )

        return conversations

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
        self._log_prompt(
            "Pass 1 (Query Generation) Response", "(N/A - Response Parsing)", content
        )
        data = self._parse_json_robust(content)
        if isinstance(data, dict):
            if "queries" in data:
                return data["queries"]
            for v in data.values():
                if isinstance(v, list):
                    return v
        return data if isinstance(data, list) else []

    def _parse_answers_output(self, content: str) -> List[Dict]:
        self._log_prompt(
            "Pass 2 (Answer Generation) Response", "(N/A - Response Parsing)", content
        )
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
