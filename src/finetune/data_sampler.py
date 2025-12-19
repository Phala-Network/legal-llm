import json
import random
import argparse
from typing import List, Dict


def sample_data(
    input_file: str, output_file: str, reject_ratio: float = 0.1, seed: int = 42
):
    """
    Samples from the training data to limit the number of 'I cannot answer' rejection samples.
    """
    random.seed(seed)

    gold_samples = []
    reject_samples = []

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check if the last assistant message is a rejection
                # High reject rate logic: "I cannot answer based on the provided results."
                last_msg = data["messages"][-1]["content"]
                if "I cannot answer based on the provided results" in last_msg:
                    reject_samples.append(data)
                else:
                    gold_samples.append(data)
            except Exception as e:
                print(f"Error parsing line: {e}")

    num_gold = len(gold_samples)
    num_reject = len(reject_samples)
    print(f"Found {num_gold} gold samples and {num_reject} rejection samples.")

    # Calculate how many rejection samples to keep
    # reject_ratio = num_reject_to_keep / (num_gold + num_reject_to_keep)
    # num_reject_to_keep = (reject_ratio * num_gold) / (1 - reject_ratio)

    if reject_ratio >= 1.0:
        keep_reject = num_reject
    else:
        keep_reject = int((reject_ratio * num_gold) / (1 - reject_ratio))
        keep_reject = min(keep_reject, num_reject)

    print(f"Keeping {keep_reject} rejection samples (Target Ratio: {reject_ratio}).")

    sampled_rejects = random.sample(reject_samples, keep_reject)
    final_data = gold_samples + sampled_rejects
    random.shuffle(final_data)

    print(f"Writing {len(final_data)} total samples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="training_data_migrated.jsonl")
    parser.add_argument("--output", type=str, default="training_data_sampled.jsonl")
    parser.add_argument(
        "--ratio", type=float, default=0.1, help="Target ratio of rejection samples"
    )
    args = parser.parse_args()

    sample_data(args.input, args.output, args.ratio)
