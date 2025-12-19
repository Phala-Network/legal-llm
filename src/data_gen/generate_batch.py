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


class BatchManager:
    def __init__(self, client):
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
            if batch.status == "completed":
                return batch.output_file_id
            elif batch.status in ["failed", "cancelled", "expired"]:
                if hasattr(batch, "errors") and batch.errors:
                    print(f"Errors: {batch.errors}")
                raise Exception(f"Batch failed: {batch.status}")
            time.sleep(poll_interval)

    def download_file(self, file_id: str, output_path: str):
        print(f"Downloading File {file_id} to {output_path}...")
        content = self.client.files.content(file_id).read()
        with open(output_path, "wb") as f:
            f.write(content)
        print("Download complete.")


class DataGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_manager = BatchManager(self.client)

    def run_pipeline(self, num_samples: int):
        pid = uuid.uuid4().hex[:8]
        print(f"Starting Automated Pipeline {pid}...")

        f1_in, f1_out = f"batch_{pid}_s1_in.jsonl", f"batch_{pid}_s1_out.jsonl"
        f2_in, f2_out = f"batch_{pid}_s2_in.jsonl", f"batch_{pid}_s2_out.jsonl"
        m1, m2 = f"batch_{pid}_m1.json", f"batch_{pid}_m2.json"

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
        json_files = glob.glob(
            os.path.join(self.data_dir, "**", "json", "*.json"), recursive=True
        )
        json_files = [
            f for f in json_files if os.path.abspath(f) not in self.processed_files
        ]
        valid_files = []
        for f in tqdm(json_files[: num_samples * 2], desc="Filtering valid cases"):
            if len(valid_files) >= num_samples:
                break
            if self._is_valid_case(f):
                valid_files.append(f)

        meta = {}
        with open(out_file, "w") as f:
            for f_path in valid_files:
                case_info = self._get_case_text(f_path)
                cid = f"req-{uuid.uuid4()}"
                body = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": self._construct_query_prompt(case_info),
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(body) + "\n")
                meta[cid] = os.path.abspath(f_path)
        with open(map_file, "w") as f:
            json.dump(meta, f, indent=2)

    def process_and_prepare_answers(
        self, in_file: str, out_file: str, m1: str, m2: str
    ):
        with open(m1, "r") as f:
            q_map = json.load(f)
        self._init_retriever()
        meta2 = {}
        with open(in_file, "r") as f_in, open(out_file, "w") as f_out:
            for line in f_in:
                res = json.loads(line)
                cid = res["custom_id"]
                if res.get("error"):
                    continue

                content = res["response"]["body"]["choices"][0]["message"]["content"]
                case_info = self._get_case_text(q_map[cid])
                queries = self._parse_queries_output(content)

                items = []
                for q_item in queries:
                    search = q_item.get("search_query")
                    context = ""
                    if search and self.retriever:
                        for i, doc in enumerate(self.retriever.retrieve(search, k=3)):
                            dcid = doc["id"].split("_")[0]
                            name = doc.get("metadata", {}).get("name", "Unknown")
                            text = (
                                self._get_full_recursive_text(doc.get("metadata", {}))
                                or doc["text"]
                            )
                            context += f"[Result {i+1}] {name} (ID: {dcid})\n{text[:3000]}...\n\n"
                    items.append(
                        {
                            "q_item": q_item,
                            "search_query": search,
                            "retrieved_context_str": context,
                        }
                    )

                ans_cid = f"ans-{uuid.uuid4()}"
                body = {
                    "custom_id": ans_cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": self._construct_answer_prompt(
                            items, case_info["text"]
                        ),
                        "response_format": {"type": "json_object"},
                    },
                }
                f_out.write(json.dumps(body) + "\n")
                meta2[ans_cid] = {"original_file": q_map[cid], "items": items}
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
                answers = self._parse_answers_output(content)

                for ans in answers:
                    idx = ans.get("item_id")
                    if idx is None or idx >= len(meta["items"]):
                        continue
                    item = meta["items"][idx]

                    msgs = [{"role": "user", "content": item["q_item"]["question"]}]

                    # Construct Assistant message (Thought + Search)
                    if item["search_query"]:
                        # User updated BaseGenerator to not use <thought> tag, so we use it directly as a "Thought" line if desired
                        # or just concatenate. For training, keeping it organized is good.
                        thought_content = ans.get("thought", "").strip()
                        # If the user wants NO <thought> tag at all, we'll just put it at the top
                        assistant_msg = f"{thought_content}\n\n<search>{item['search_query']}</search>"
                        msgs.append({"role": "assistant", "content": assistant_msg})
                        msgs.append(
                            {
                                "role": "user",
                                "content": f"Search Results:\n{item['retrieved_context_str']}",
                            }
                        )

                    msgs.append({"role": "assistant", "content": ans.get("answer")})
                    f_out.write(json.dumps({"messages": msgs}) + "\n")
                    count += 1

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

    args = parser.parse_args()
    gen = DataGenerator(output_file=args.output_file)

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
