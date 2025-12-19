import os
import json
import random
import glob
import argparse
from typing import List, Dict, Optional
from tqdm import tqdm
import time
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_gen.base_generator import BaseGenerator


class DataGenerator(BaseGenerator):
    def run(self, num_samples: int = 10):
        self._init_retriever()
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

        json_files = [
            f for f in json_files if os.path.abspath(f) not in self.processed_files
        ]
        print(f"Remaining candidates after filtering history: {len(json_files)}")

        random.shuffle(json_files)

        valid_count = 0
        pbar = tqdm(total=num_samples, desc="Generating Data")
        file_idx = 0
        while valid_count < num_samples and file_idx < len(json_files):
            json_file = json_files[file_idx]
            file_idx += 1

            case_info = self._get_case_text(json_file)
            prompt = self._construct_query_prompt(case_info)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                queries = self._parse_queries_output(
                    response.choices[0].message.content
                )
                if not queries:
                    continue

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
                            context_str += f"[Result {i+1}] {name} (ID: {cid})\n{text[:3000]}...\n\n"

                    items_to_process.append(
                        {
                            "q_item": q_item,
                            "search_query": search_query,
                            "retrieved_context_str": context_str,
                        }
                    )

                ans_prompt = self._construct_answer_prompt(
                    items_to_process, case_info["text"]
                )
                ans_res = self.client.chat.completions.create(
                    model=self.model,
                    messages=ans_prompt,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                answers = self._parse_answers_output(ans_res.choices[0].message.content)

                for ans_data in answers:
                    idx = ans_data.get("item_id")
                    if idx is None or idx >= len(items_to_process):
                        continue
                    item = items_to_process[idx]

                    messages = [{"role": "user", "content": item["q_item"]["question"]}]
                    if item["search_query"]:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"<thought>{ans_data.get('thought')}</thought>\n<search>{item['search_query']}</search>",
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Search Results:\n{item['retrieved_context_str']}",
                            }
                        )
                    messages.append(
                        {"role": "assistant", "content": ans_data.get("answer")}
                    )

                    with open(self.output_file, "a", encoding="utf-8") as out_f:
                        out_f.write(json.dumps({"messages": messages}) + "\n")
                    valid_count += 1
                    pbar.update(1)

                with open(self.processed_log, "a") as f:
                    f.write(os.path.abspath(json_file) + "\n")
                self.processed_files.add(os.path.abspath(json_file))
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        pbar.close()
        print(f"Data Generation Complete. Generated {valid_count} examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    generator = DataGenerator()
    generator.run(num_samples=args.num_samples)
