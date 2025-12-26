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
        json_files = self.get_valid_case_files(num_samples)

        if not json_files:
            print("No valid candidates found.")
            return

        print(f"Generating data for {len(json_files)} files...")

        valid_count = 0
        pbar = tqdm(total=num_samples, desc="Generating Data")

        for json_file in json_files:
            if valid_count >= num_samples:
                break

            case_info = self._get_case_text(json_file)
            prompt, strategy = self._construct_query_prompt(case_info)

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

                items_to_process = self.augment_queries_with_context(queries)

                # Inject strategy for Pass 2
                for item in items_to_process:
                    item["q_item"]["cot_strategy_name"] = strategy

                conversations = self.construct_answer_conversations(
                    items_to_process, case_info["text"]
                )

                # Process each conversation individually (Pass 2)
                for i, msgs in enumerate(conversations):
                    if not msgs:
                        continue

                    try:
                        ans_res = self.client.chat.completions.create(
                            model=self.model,
                            messages=msgs,
                            response_format={"type": "json_object"},
                            temperature=0.3,
                        )
                        ans_content = ans_res.choices[0].message.content
                        ans_data = self._parse_json_robust(ans_content)

                        if not ans_data:
                            continue

                        # items_to_process[i] corresponds to conversations[i]
                        item = items_to_process[i]
                        final_messages = self.construct_final_messages(item, ans_data)

                        with open(self.output_file, "a", encoding="utf-8") as out_f:
                            out_f.write(json.dumps({"messages": final_messages}) + "\n")

                        valid_count += 1
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error processing item {i} in {json_file}: {e}")

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
