import os
import json
import glob
import re
from tqdm import tqdm


def migrate_training_data(
    data_dir="data",
    input_file="training_data.jsonl",
    output_file="training_data_migrated.jsonl",
):
    print("Building Path-to-ID Map...")
    path_to_id = {}
    json_files = glob.glob(
        os.path.join(data_dir, "**", "json", "*.json"), recursive=True
    )

    for fpath in tqdm(json_files, desc="Scanning files"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                cid = str(data.get("id"))
                if cid:
                    # Map both full path and relative path for robustness
                    rel_path = os.path.relpath(fpath, os.getcwd())
                    path_to_id[fpath] = cid
                    path_to_id[rel_path] = cid
                    # Also map just the filename part or subdir part if used in citations
                    # (e.g., data/cal-rptr-3d/226/json/0045-01.json)
                    path_to_id[fpath.replace(os.path.abspath(data_dir), "data")] = cid
        except Exception as e:
            pass

    print(f"Mapped {len(path_to_id)} paths to IDs.")

    print(f"Migrating {input_file}...")
    with (
        open(input_file, "r", encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):

        for line in tqdm(f_in, desc="Processing lines"):
            try:
                msg_data = json.loads(line)
                for msg in msg_data["messages"]:
                    content = msg["content"]

                    # Pattern 1: (File: path/to/case.json)
                    matches_file = re.findall(r"\(File: (.*?)\)", content)
                    for path in matches_file:
                        clean_path = path.strip()
                        if clean_path in path_to_id:
                            content = content.replace(
                                f"(File: {path})", f"(ID: {path_to_id[clean_path]})"
                            )
                        else:
                            basename = os.path.basename(clean_path)
                            for k, v in path_to_id.items():
                                if k.endswith(basename):
                                    content = content.replace(
                                        f"(File: {path})", f"(ID: {v})"
                                    )
                                    break

                    # Pattern 2: [Name](path/to/case.json)
                    matches_link = re.findall(r"\[.*?\]\((.*?)\)", content)
                    for path in matches_link:
                        clean_path = path.strip()
                        # Only replace if it looks like a path we know
                        if clean_path in path_to_id:
                            content = content.replace(
                                f"({path})", f"(ID: {path_to_id[clean_path]})"
                            )
                        else:
                            basename = os.path.basename(clean_path)
                            if basename.endswith(".json"):
                                for k, v in path_to_id.items():
                                    if k.endswith(basename):
                                        content = content.replace(
                                            f"({path})", f"(ID: {v})"
                                        )
                                        break

                    msg["content"] = content

                f_out.write(json.dumps(msg_data) + "\n")
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Migration complete. New file: {output_file}")


if __name__ == "__main__":
    migrate_training_data()
