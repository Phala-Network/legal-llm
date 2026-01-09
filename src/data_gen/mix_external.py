import json
import argparse
import os


def mix_slimorca(output_file: str, num_samples: int = 1000):
    """
    Loads samples from SlimOrca, converts them to the local format,
    and appends them to the output file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Please run with 'uv run'.")
        return

    print(f"Loading {num_samples} samples from SlimOrca...")
    try:
        # Use streaming=True to avoid downloading the whole dataset
        ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    count = 0
    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "a") as f_out:
        for example in ds.take(num_samples):
            messages = []
            for msg in example["conversations"]:
                role = "user" if msg["from"] == "human" else "assistant"
                if msg["from"] == "system":
                    role = "system"
                messages.append({"role": role, "content": msg["value"]})

            f_out.write(json.dumps({"messages": messages}) + "\n")
            count += 1
    print(f"Successfully mixed in {count} SlimOrca samples into {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mix external datasets into training data."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Target JSONL file to append data to",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="slimorca",
        choices=["slimorca"],
        help="Dataset to mix in",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to mix in"
    )

    args = parser.parse_args()

    if args.dataset == "slimorca":
        mix_slimorca(args.output_file, args.num_samples)
