import os
import json
import random
import glob
import argparse
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import time
import uuid
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_gen.base_generator import BaseGenerator
from src.utils.path_utils import normalize_case_path


from src.utils.batch_utils import BatchManager


class DataGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_manager = BatchManager(self.client)
        self.rag_params = {
            "db_path": kwargs.get("db_path", "chroma_db"),
            "index_dir": kwargs.get("index_dir", "tantivy_index"),
            "shard_assignments": kwargs.get(
                "shard_assignments", "data/shard_assignments.json"
            ),
        }

    def run_pipeline(self, num_samples: int):
        pid = uuid.uuid4().hex[:8]
        print(f"Starting Automated Pipeline {pid}...")

        f1_in = os.path.join(self.output_dir, f"batch_{pid}_s1_in.jsonl")
        f1_out = os.path.join(self.output_dir, f"batch_{pid}_s1_out.jsonl")
        f2_in = os.path.join(self.output_dir, f"batch_{pid}_s2_in.jsonl")
        f2_out = os.path.join(self.output_dir, f"batch_{pid}_s2_out.jsonl")
        m1 = os.path.join(self.output_dir, f"batch_{pid}_m1.json")
        m2 = os.path.join(self.output_dir, f"batch_{pid}_m2.json")

        # Stage 1: Queries
        print("\n=== STAGE 1: Generating Queries ===")
        self.prepare_batch_queries(num_samples, f1_in, m1)
        fid1 = self.batch_manager.upload_file(f1_in)
        bid1 = self.batch_manager.create_batch(fid1, f"S1-{pid}")
        out_fid1 = self.batch_manager.wait_for_batch(bid1)
        self.batch_manager.download_file(out_fid1, f1_out)

        # Stage 2: Answers
        print("\n=== STAGE 2: Running RAG & Preparing Answers ===")
        self.process_and_prepare_answers(f1_out, f2_in, m1, m2)
        fid2 = self.batch_manager.upload_file(f2_in)
        bid2 = self.batch_manager.create_batch(fid2, f"S2-{pid}")
        out_fid2 = self.batch_manager.wait_for_batch(bid2)
        self.batch_manager.download_file(out_fid2, f2_out)

        # Stage 3: Finalize
        print("\n=== STAGE 3: Finalizing Dataset ===")
        self.finalize_dataset(f2_out, m2)
        print(f"\nPipeline {pid} Complete! Data in: {self.output_file}")

    def prepare_batch_queries(self, num_samples: int, out_file: str, map_file: str):
        valid_files = self.get_valid_case_files(num_samples)

        meta = {}
        with open(out_file, "w") as f:
            for f_path in valid_files:
                case_info = self._get_case_text(f_path)
                cid = f"req-{uuid.uuid4()}"
                messages, strategy = self._construct_query_prompt(case_info)
                body = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(body) + "\n")
                meta[cid] = {"path": os.path.abspath(f_path), "strategy": strategy}
        with open(map_file, "w") as f:
            json.dump(meta, f, indent=2)

    def process_and_prepare_answers(
        self, in_file: str, out_file: str, m1: str, m2: str
    ):
        with open(m1, "r") as f:
            q_map = json.load(f)
        self._init_retriever(**self.rag_params)
        meta2 = {}
        with open(in_file, "r") as f_in, open(out_file, "w") as f_out:
            for line in f_in:
                res = json.loads(line)
                cid = res["custom_id"]
                if res.get("error"):
                    continue

                content = res["response"]["body"]["choices"][0]["message"]["content"]

                # Meta is now a dict {path: ..., strategy: ...}
                case_path = q_map[cid]["path"]
                case_info = self._get_case_text(case_path)
                strategy = q_map[cid]["strategy"]

                # For focus_case_id we need normalized path relative to data_dir
                # BaseGenerator keeps data_dir in self.data_dir
                rel_path = os.path.relpath(case_path, self.data_dir)
                norm_path = normalize_case_path(rel_path)

                queries = self._parse_queries_output(content)
                items = self.augment_queries_with_context(
                    queries, focus_case_id=norm_path
                )

                # Prepare Stage 2 requests (Fan-Out)
                # Call construct_answer_conversations which returns List[List[Dict]] (list of conversations)

                # First, ensure we inject the strategy name into items so construct_answer_conversations can see it
                for item in items:
                    item["q_item"]["cot_strategy_name"] = strategy

                conversations = self.construct_answer_conversations(
                    items, case_info["text"]
                )

                for i, messages in enumerate(conversations):
                    if messages is None:
                        continue

                    ans_cid = f"ans-{uuid.uuid4()}"
                    body = {
                        "custom_id": ans_cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": messages,
                            "response_format": {"type": "json_object"},
                        },
                    }
                    f_out.write(json.dumps(body) + "\n")

                    # We need to link this specific answer request back to the specific item
                    # items[i] corresponds to conversations[i] because of order preservation
                    meta2[ans_cid] = {
                        "original_file": q_map[cid]["path"],
                        "item": items[i],
                    }
        with open(m2, "w") as f:
            json.dump(meta2, f, indent=2)

    def finalize_dataset(self, in_file: str, m2: str):
        with open(m2, "r") as f:
            a_map = json.load(f)
        count = 0
        with (
            open(in_file, "r") as f_in,
            open(self.output_file, "a") as f_out,
            open(self.processed_log, "a") as f_log,
        ):
            for line in f_in:
                res = json.loads(line)
                cid = res["custom_id"]
                if res.get("error"):
                    continue

                meta = a_map[cid]
                content = res["response"]["body"]["choices"][0]["message"]["content"]

                # In the new architecture, each request corresponds to ONE item.
                # The content is coverage JSON: {thought: ..., answer: ...}

                ans_data = self._parse_json_robust(content)
                if not ans_data:
                    continue

                item = meta["item"]
                msgs = self.construct_final_messages(item, ans_data)

                f_out.write(json.dumps({"messages": msgs}) + "\n")
                count += 1

                # We don't want to log the same file too many times, but simpler to preserve existing behavior or optimize?
                # The original code wrote to log for every ANSWER.
                # We'll just write it. deduplication happens at load time.
                f_log.write(os.path.abspath(meta["original_file"]) + "\n")
        print(f"Added {count} examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", action="store_true", help="Run full automated pipeline"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["queries", "answers", "finalize"],
        help="Manual processing stage",
    )
    parser.add_argument("--num_samples", type=int, default=10)
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
    parser.add_argument(
        "--db_path", type=str, default="chroma_db", help="Chroma DB path"
    )
    parser.add_argument(
        "--index_dir", type=str, default="tantivy_index", help="Tantivy index directory"
    )
    parser.add_argument(
        "--shard_assignments",
        type=str,
        default="data/shard_assignments.json",
        help="Shard assignments JSON path",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Cases data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory for all output files"
    )

    args = parser.parse_args()
    gen = DataGenerator(
        output_file=args.output_file,
        output_dir=args.output_dir,
        db_path=args.db_path,
        index_dir=args.index_dir,
        shard_assignments=args.shard_assignments,
        data_dir=args.data_dir,
    )

    if args.pipeline:
        gen.run_pipeline(args.num_samples)
    elif args.stage == "queries":
        gen.prepare_batch_queries(
            num_samples=args.num_samples,
            map_file=args.map_file,
            out_file=args.output_file,
        )
    elif args.stage == "answers":
        if not args.input_file:
            print("Error: --input_file required")
        else:
            gen.process_and_prepare_answers(
                in_file=args.input_file,
                out_file=args.output_file,
                m1=args.map_file,
                m2=args.next_map_file,
            )
    elif args.stage == "finalize":
        if not args.input_file:
            print("Error: --input_file required")
        else:
            gen.finalize_dataset(in_file=args.input_file, m2=args.map_file)
    else:
        print("Please specify --pipeline or --stage.")
