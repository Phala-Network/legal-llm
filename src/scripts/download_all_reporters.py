import os
import json
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from download_reporter import download_file, download_single_reporter

REPORTERS_METADATA_URL = "https://static.case.law/ReportersMetadata.json"
VOLUMES_METADATA_URL = "https://static.case.law/VolumesMetadata.json"
JURISDICTIONS_METADATA_URL = "https://static.case.law/JurisdictionsMetadata.json"


def process_reporter_wrapper(args):
    """Wrapper for download_single_reporter to be used in multiprocess"""
    slug, output_dir, max_volumes, delay, show_progress = args
    download_single_reporter(slug, output_dir, max_volumes, delay, show_progress)
    return slug


def main():
    parser = argparse.ArgumentParser(
        description="Download all reporters and metadata from static.case.law"
    )
    parser.add_argument("--output_dir", default="data", help="Root directory for data")
    parser.add_argument(
        "--max_reporters",
        type=int,
        default=0,
        help="Max reporters to process (0 for all)",
    )
    parser.add_argument(
        "--max_volumes",
        type=int,
        default=0,
        help="Max volumes to download per reporter (0 for all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between downloads (default: 1.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel downloading (default: 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download Root Metadata
    print("Downloading root metadata...")
    root_metadata_files = [
        (REPORTERS_METADATA_URL, "ReportersMetadata.json"),
        (VOLUMES_METADATA_URL, "VolumesMetadata.json"),
        (JURISDICTIONS_METADATA_URL, "JurisdictionsMetadata.json"),
    ]

    for url, filename in root_metadata_files:
        output_path = os.path.join(args.output_dir, filename)
        print(f"Downloading {filename}...")
        download_file(url, output_path)
        if args.delay > 0:
            time.sleep(args.delay)

    # 2. Parse Reporters Metadata
    reporters_metadata_path = os.path.join(args.output_dir, "ReportersMetadata.json")
    try:
        with open(reporters_metadata_path, "r") as f:
            reporters = json.load(f)
    except Exception as e:
        print(f"Error parsing ReportersMetadata.json: {e}")
        return

    print(f"Found {len(reporters)} reporters.")

    # 3. Filter Reporters
    reporters_to_process = []
    count = 0
    for reporter in reporters:
        if args.max_reporters > 0 and count >= args.max_reporters:
            break
        slug = reporter.get("slug")
        if not slug:
            continue
        reporters_to_process.append(slug)
        count += 1

    print(
        f"Processing {len(reporters_to_process)} reporters with {args.workers} workers..."
    )

    # 4. Process Reporters (Parallel or Sequential)
    if args.workers > 1:
        # Multi-processing
        # Hide individual progress bars in parallel mode to keep output clean
        show_child_progress = False

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for slug in reporters_to_process:
                # Pack arguments for the wrapper
                # Note: delay is per-process, so it helps rate limit individual workers,
                # but valid total rate will be delay/workers roughly
                task_args = (
                    slug,
                    args.output_dir,
                    args.max_volumes,
                    args.delay,
                    show_child_progress,
                )
                futures.append(executor.submit(process_reporter_wrapper, task_args))

            # Use tqdm to show overall completion progress
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Reporters"):
                pass
    else:
        # Sequential processing
        # Show individual progress bars
        show_child_progress = True

        for i, slug in enumerate(reporters_to_process):
            print(f"Processing reporter: {slug} ({i + 1}/{len(reporters_to_process)})")
            download_single_reporter(
                slug, args.output_dir, args.max_volumes, args.delay, show_child_progress
            )
            if args.delay > 0:
                time.sleep(args.delay)

    print("All requested reporters processed.")


if __name__ == "__main__":
    main()
