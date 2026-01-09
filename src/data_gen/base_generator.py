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
        output_dir: str = ".",
        model: Optional[str] = None,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # If output_file is just a name, put it in output_dir
        if not os.path.dirname(output_file) and output_dir != ".":
            self.output_file = os.path.join(self.output_dir, output_file)
        else:
            self.output_file = output_file

        self.processed_log = os.path.join(self.output_dir, "processed_files.txt")
        self.processed_files = set()
        if os.path.exists(self.processed_log):
            with open(self.processed_log, "r") as f:
                self.processed_files = set(f.read().splitlines())

        self.client = OpenAI()
        self.model = model or os.getenv("GENERATION_MODEL", "gpt-4o")
        self.parser = CaseParser(data_dir=data_dir)
        self.retriever = None
        self.selection_counter = 0

    def _log_prompt(self, stage: str, prompt: str, response: str):
        """
        Logs the prompt and response to a debug file for inspection.
        """
        log_file = os.path.join(self.output_dir, "generation_debug.log")
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

    def _init_retriever(
        self,
        db_path="chroma_db",
        index_dir="tantivy_index",
        shard_assignments="data/shard_assignments.json",
    ):
        if self.retriever:
            return

        print(
            f"Initializing CaseRetriever (DB: {db_path}, Index: {index_dir}, Assignments: {shard_assignments})..."
        )
        try:
            self.retriever = CaseRetriever(
                db_path=db_path,
                index_dir=index_dir,
                shard_assignments=shard_assignments,
            )
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

    def _parse_structured_query(self, query_str: str) -> Dict[str, str]:
        """
        Parses structured queries like "KEYWORDS: ...; TIME: ...; COURT: ...; JURISDICTION: ..."
        If no structure is found, treats the entire string as keywords.
        """
        params = {
            "keywords": None,
            "time": None,
            "court": None,
            "jurisdiction": None,
        }

        if not query_str:
            return params

        parts = query_str.split(";")
        found_any_key = False

        # Check if it's structured at all (has any of the known keys)
        known_keys = {"KEYWORDS", "TIME", "COURT", "JURISDICTION"}

        for part in parts:
            if ":" in part:
                key = part.split(":", 1)[0].strip().upper()
                if key in known_keys:
                    found_any_key = True
                    break

        if not found_any_key:
            params["keywords"] = query_str.strip()
            return params

        for part in parts:
            if ":" in part:
                key, val = part.split(":", 1)
                key = key.strip().upper()
                val = val.strip()
                if key == "KEYWORDS":
                    params["keywords"] = val
                elif key == "TIME":
                    params["time"] = val
                elif key == "COURT":
                    params["court"] = val
                elif key == "JURISDICTION":
                    params["jurisdiction"] = val

        return params

    def augment_queries_with_context(
        self, queries: List[Dict], focus_case_id: str = None
    ) -> List[Dict]:
        """
        Takes a list of query items (from LLM output), performs retrieval if search_query is present,
        and returns a list of items ready for answer generation.
        """
        items_to_process = []
        for q_item in queries:
            search_query_raw = q_item.get("search_query")
            context_str = ""

            if search_query_raw and self.retriever:
                parsed_params = self._parse_structured_query(search_query_raw)
                search_query = parsed_params["keywords"]

                print(
                    f"Retrieving for query: {search_query} (Focus Case: {focus_case_id})"
                )
                retrieved = self.retriever.retrieve(
                    search_query, k=4, focus_case_id=focus_case_id
                )

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
                    "search_query": search_query_raw,
                    "retrieved_context_str": context_str,
                }
            )
        return items_to_process

    def _construct_query_prompt(self, case_info: Dict) -> List[Dict]:
        context_text = case_info.get("text", "")
        case_id = case_info.get("id", "unknown")
        case_name = case_info.get("name", "Unknown Case")

        # Select style and COT using a counter to ensure even distribution
        # one selection_counter is enough since we have 4 QUERY_STYLES and 7 COT_STRATEGIES,
        # so this will exhaust all combinations
        selected_style = self.QUERY_STYLES[
            self.selection_counter % len(self.QUERY_STYLES)
        ]
        selected_cot = self.COT_STRATEGIES[
            self.selection_counter % len(self.COT_STRATEGIES)
        ]
        self.selection_counter += 1

        distribution = self._get_random_query_distribution()

        prompt = f"""
        Your persona: {selected_style['style_name']}
        {selected_style['instruction']}

        Task: Create a diverse set of training queries based on the provided legal text.

        Reasoning Strategy: {selected_cot['prompt']}

        Target Distribution for this batch:
        1. [COMPLEX] * {distribution['complex']}: Multi-step reasoning.
        2. [SIMPLE] * {distribution['simple']}: Specific fact retrieval.
        3. [GENERAL] * {distribution['general']}: General legal concepts.
        4. [NEGATIVE] * {distribution['negative']}: Out-of-scope.

        Output Structure (JSON ONLY):
        {{
            "queries": [
                {{
                    "type": "complex",
                    "thought": "<thought>Initial analysis of jurisdiction...</thought><thought>Secondary check on standing requirements...</thought>",
                    "question": "...",
                    "search_query": "..."
                }},
                {{
                    "type": "complex",
                    "thought": "Thought explaining why specific keywords and optional filters (like date ranges or court levels) are chosen...",
                    "question": "...",
                    "search_query": "KEYWORDS: ...; (optional) TIME : ...; (optional) COURT: ...; (optional) JURISDICTION: ..."
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
        {
            "name": "Search_Keyword_Extraction",
            "prompt": "Strategy: Professional Search Extraction. In the 'thought' field, first identify the core legal entities, then the specific cause of action, then narrow down the relevant timeline and jurisdiction if identifiable. Finally, combine these into a structured search query: 'KEYWORDS: ...; TIME: ...; COURT: ...; JURISDICTION: ...'. Note: TIME, COURT, and JURISDICTION are optional; omit them if they cannot be determined from the context.",
        },
        {
            "name": "Interactive_Search",
            "prompt": "Strategy: Client-Consultant Interaction. In the 'thought' field, reason about why you need to propose the search parameters to the user first. Propose the keywords and any relevant filters (Time, Court, Jurisdiction) if identifiable, explaining why they are relevant to the case. The model must present this as a proposal: 'I suggest searching for [Keywords] with [Filters]. Does this cover the scope of your inquiry?'",
        },
    ]

    def _get_random_query_distribution(self):
        # Randomize the mix to avoid fixed patterns, but ensure negative_q is always 1
        total = 15
        negative_q = 1

        # Distributed remaining 14 among complex, simple, general
        remaining = total - negative_q

        complex_q = random.randint(6, 9)
        remaining -= complex_q

        simple_q = random.randint(1, remaining - 1)
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
                conversations.append(None)
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

            assistant_content = self._format_thought(
                q_item.get("thought", "Thinking...")
            )
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
        return {}

    def _format_thought(self, thought_text: str) -> str:
        """
        Ensures the thought is properly wrapped in <thought> tags.
        Supports multiple <thought> blocks if already present.
        """
        if not thought_text:
            return "<thought>Thinking...</thought>"
        thought_text = thought_text.strip()
        if thought_text.startswith("<thought>"):
            return thought_text
        return f"<thought>{thought_text}</thought>"

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

    def construct_final_messages(self, item: Dict, ans_data: Dict) -> List[Dict]:
        """
        Constructs the final list of messages for fine-tuning.
        item: dict containing 'q_item' (question), 'search_query', 'retrieved_context_str'
        ans_data: dict containing 'thought' and 'answer'
        """
        q_item = item["q_item"]
        strategy = q_item.get("cot_strategy_name", "")
        msgs = [{"role": "user", "content": q_item["question"]}]

        # Pass 1 Thought (Stage 1 Reasoning)
        p1_thought = self._format_thought(q_item.get("thought", ""))

        # Pass 2 Thought (Stage 2 Reasoning)
        p2_thought = self._format_thought(ans_data.get("thought", ""))

        search_query = item.get("search_query")

        if strategy == "Interactive_Search" and search_query:
            # Multi-turn interaction simulation
            # 1. Model proposes search
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"{p1_thought}\nI've analyzed your request. I propose searching for the following parameters:\n{search_query}\n\nShall I proceed with this search?",
                }
            )

            # 2. User confirms or slightly adjusts (Simulated)
            confirmation_choice = random.random()
            if confirmation_choice < 0.7:
                user_feedback = "Yes, please proceed."
            else:
                user_feedback = "That looks good, but also include any mentions of 'pre-existing conditions' if applicable."
                # We don't actually re-run search here for the simulation consistency,
                # but we show the model's ability to take feedback.

            msgs.append({"role": "user", "content": user_feedback})

            # 3. Model acknowledges and shows search
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"<thought>The user confirmed/refined the search. Executing now.</thought>\n<search>{search_query}</search>",
                }
            )

            # 4. Search Results
            msgs.append(
                {
                    "role": "user",
                    "content": f"Search Results:\n{item['retrieved_context_str']}",
                }
            )

            # 5. Final Answer
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"{p2_thought}\n{ans_data.get('answer')}",
                }
            )

        elif search_query:
            # Standard Multi-step with search
            # We use Stage 1 thought to justify the search
            assistant_msg = f"{p1_thought}\n<search>{search_query}</search>"
            msgs.append({"role": "assistant", "content": assistant_msg})

            msgs.append(
                {
                    "role": "user",
                    "content": f"Search Results:\n{item.get('retrieved_context_str', 'No results found.')}",
                }
            )
            # We use Stage 2 thought for the final derivation
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"{p2_thought}\n{ans_data.get('answer')}",
                }
            )
        else:
            # Direct answer with thought
            # In this case, there's only one stage usually, but if two stages were used:
            combined_thought = p1_thought
            if p2_thought:
                combined_thought += f"\n{p2_thought}"

            msgs.append(
                {
                    "role": "assistant",
                    "content": f"{combined_thought}\n{ans_data.get('answer')}",
                }
            )
        return msgs
