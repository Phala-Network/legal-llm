import os
import json
import shutil
import random
from typing import List, Dict, Set, Optional
import sys
import argparse

# Add src to path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from rag.case_filtering import get_cited_case_paths, filter_cases_by_date


def get_recent_reporters(
    data_dir: str, since_year: Optional[int] = None
) -> tuple[Dict[str, List[str]], int]:
    """
    Parses JurisdictionsMetadata.json to find reporters with data >= start_year.
    If since_year is None, it finds the latest year in metadata and uses that.
    Returns a mapping of jurisdiction_name to a list of reporter_slugs and the actual start_year used.
    """
    metadata_path = os.path.join(data_dir, "JurisdictionsMetadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return {}, 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        jurisdictions = json.load(f)

    # Find max end_year among EXISTING reporters
    max_year = 0
    for jur in jurisdictions:
        for rep in jur.get("reporters", []):
            slug = rep.get("slug")
            if slug and os.path.exists(os.path.join(data_dir, slug)):
                ey = rep.get("end_year")
                if ey and ey > max_year:
                    max_year = ey

    if max_year == 0:
        print("No existing reporters found on disk.")
        return {}, 0

    if since_year is None:
        since_year = max_year - 2  # Default to last 3 years

    print(f"Targeting data from year {since_year} onwards.")

    result = {}
    for jur in jurisdictions:
        jur_name = jur.get("name_long") or jur.get("name")
        valid_reporters = []
        for rep in jur.get("reporters", []):
            if rep.get("end_year") and rep.get("end_year") >= since_year:
                slug = rep.get("slug")
                if slug and os.path.exists(os.path.join(data_dir, slug)):
                    valid_reporters.append(slug)

        if valid_reporters:
            result[jur_name] = valid_reporters

    return result, since_year


def sample(
    data_dir: str,
    output_dir: str,
    num_samples: int = 100,
    since_year: Optional[int] = None,
    min_length: int = 3000,
):
    """
    Samples cases using filter_cases_by_date and ensuring jurisdiction diversity.
    """
    # 1. Map jurisdictions to recent reporters
    jur_to_reporters, start_year = get_recent_reporters(data_dir, since_year)
    if not jur_to_reporters:
        print("No recent reporters found.")
        return

    start_date_str = f"{start_year}-01-01"
    jurisdictions = list(jur_to_reporters.keys())
    random.shuffle(jurisdictions)

    samples_per_jur = max(
        1, (num_samples + len(jurisdictions) - 1) // len(jurisdictions)
    )

    root_cases_found = []

    # 3. Targeted scan using filter_cases_by_date
    print(f"Starting sample process in {len(jurisdictions)} jurisdictions...")

    for jur in jurisdictions:
        if len(root_cases_found) >= num_samples:
            break

        reporters = jur_to_reporters[jur]
        jur_samples_collected = 0

        random.shuffle(reporters)
        for slug in reporters:
            if (
                jur_samples_collected >= samples_per_jur
                or len(root_cases_found) >= num_samples
            ):
                break

            rep_path = os.path.join(data_dir, slug)

            # REUSE logic: Call filter_cases_by_date for this specific reporter directory
            candidate_files = filter_cases_by_date(
                rep_path, start_date=start_date_str, min_content_length=min_length
            )

            if not candidate_files:
                continue

            # Randomly pick from matches in this reporter
            num_to_take = min(
                len(candidate_files), samples_per_jur - jur_samples_collected
            )
            # Ensure we don't exceed total num_samples
            num_to_take = min(num_to_take, num_samples - len(root_cases_found))

            samples = random.sample(candidate_files, num_to_take)
            root_cases_found.extend(samples)
            jur_samples_collected += len(samples)

            for s in samples:
                print(
                    f"[{len(root_cases_found)}/{num_samples}] Found case in {jur}: {os.path.relpath(s, data_dir)}"
                )

    # 4. Copy and Resolve Citations
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copied_files = set()
    root_case_rel_paths = []

    def copy_file(abs_src: str):
        rel = os.path.relpath(abs_src, data_dir)
        if abs_src in copied_files:
            return rel
        dest = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(abs_src, dest)
        copied_files.add(abs_src)
        return rel

    print("\nCopying files and resolving citations...")
    for i, root_path in enumerate(root_cases_found):
        rel = copy_file(root_path)
        root_case_rel_paths.append(rel)

        try:
            with open(root_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            cited_files = get_cited_case_paths(case_data, data_dir)
            for c in cited_files:
                if os.path.exists(c):
                    copy_file(c)
        except Exception as e:
            print(f"Error resolving citations for {root_path}: {e}")

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(root_cases_found)} root cases...")

    # 5. Save Metadata
    metadata_path = os.path.join(output_dir, "root_cases.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(root_case_rel_paths, f, indent=2)

    print(f"\nDone! Copied {len(copied_files)} total files to {output_dir}")
    print(f"Root cases list saved to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample cases with jurisdiction diversity and content length checks."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of root cases to sample."
    )
    parser.add_argument(
        "--since_year", type=int, default=None, help="Start year for sampling."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=3000,
        help="Minimum characters in opinions text.",
    )
    parser.add_argument(
        "--output", type=str, default="data_sample", help="Output directory."
    )

    args = parser.parse_args()

    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, args.output)

    sample(
        DATA_ROOT,
        OUTPUT_ROOT,
        num_samples=args.num_samples,
        since_year=args.since_year,
        min_length=args.min_length,
    )
