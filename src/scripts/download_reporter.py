import os
import json
import requests
import argparse
from tqdm import tqdm
import zipfile
import shutil
import time

METADATA_URL_TEMPLATE = "https://static.case.law/{reporter}/VolumesMetadata.json"
REPORTER_METADATA_URL_TEMPLATE = (
    "https://static.case.law/{reporter}/ReporterMetadata.json"
)
CASES_METADATA_URL_TEMPLATE = "https://static.case.law/{reporter}/CasesMetadata.json"
VOLUME_URL_TEMPLATE = "https://static.case.law/{reporter}/{volume_number}.zip"


def download_file(url, output_path, show_progress=True):
    """Downloads a file with a progress bar."""
    try:
        response = requests.get(url, stream=True)

        if response.status_code == 404:
            if show_progress:
                print(f"{os.path.basename(output_path)} does not exist.")
            return False

        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        if show_progress:
            with (
                open(output_path, "wb") as f,
                tqdm(
                    desc=os.path.basename(output_path),
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    bar.update(size)
        else:
            with open(output_path, "wb") as f:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def unzip_file(zip_path, extract_to):
    """Unzips a file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")
        return False


def download_single_reporter(
    reporter, output_dir="data", max_volumes=0, delay=1.0, show_progress=True
):
    """Downloads a single reporter's metadata and volumes.

    Args:
        reporter: The slug of the reporter (e.g., 'cal-rptr-3d').
        output_dir: The directory to save data to.
        max_volumes: Maximum number of volumes to download (0 for all).
        delay: Time in seconds to sleep between downloads to allow being polite to the server.
        show_progress: Whether to show progress bars for file downloads.
    """
    reporter_dir = os.path.join(output_dir, reporter)
    os.makedirs(reporter_dir, exist_ok=True)

    if show_progress:
        print(f"Setting up for {reporter} in {reporter_dir}")

    # 1. Download Metadata
    files_to_download = [
        (METADATA_URL_TEMPLATE.format(reporter=reporter), "VolumesMetadata.json"),
        (
            REPORTER_METADATA_URL_TEMPLATE.format(reporter=reporter),
            "ReporterMetadata.json",
        ),
        (CASES_METADATA_URL_TEMPLATE.format(reporter=reporter), "CasesMetadata.json"),
    ]

    for url, filename in files_to_download:
        output_path = os.path.join(reporter_dir, filename)
        if show_progress:
            print(f"Downloading {filename}...")
        download_file(
            url, output_path, show_progress=show_progress
        )  # We don't exit on failure here, as some might be optional or fail individually
        if delay > 0:
            time.sleep(delay)

    # 2. Parse Metadata (specifically VolumesMetadata.json for volumes list)
    volumes_metadata_path = os.path.join(reporter_dir, "VolumesMetadata.json")
    if not os.path.exists(volumes_metadata_path):
        if show_progress:
            print(
                "VolumesMetadata.json not found. Cannot proceed with volumes download."
            )
        return

    try:
        with open(volumes_metadata_path, "r") as f:
            volumes = json.load(f)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return

    if show_progress:
        print(f"Found {len(volumes)} volumes.")

    # 3. Download Volumes
    count = 0
    for vol in volumes:
        if max_volumes > 0 and count >= max_volumes:
            if show_progress:
                print(f"Reached max volumes limit ({max_volumes}). Stopping.")
            break

        vol_num = vol.get("volume_number")
        if not vol_num:
            continue

        if show_progress:
            print(f"Processing Volume {vol_num}...")

        # Define paths
        zip_name = f"{vol_num}.zip"
        zip_path = os.path.join(reporter_dir, zip_name)
        extract_dir = os.path.join(reporter_dir, vol_num)

        # Check if already exists (simple check: if directory exists and not empty)
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            if show_progress:
                print(f"  Volume {vol_num} already appears to be downloaded. Skipping.")
            count += 1
            continue

        # Download
        url = VOLUME_URL_TEMPLATE.format(reporter=reporter, volume_number=vol_num)
        if download_file(url, zip_path, show_progress=show_progress):
            # Unzip
            if show_progress:
                print(f"  Unzipping to {extract_dir}...")
            os.makedirs(extract_dir, exist_ok=True)
            if unzip_file(zip_path, extract_dir):
                if show_progress:
                    print(f"  Unzip successful.")
                # Cleanup Zip
                os.remove(zip_path)
                count += 1
                if delay > 0:
                    time.sleep(delay)
            else:
                print(f"  Unzip failed.")
        else:
            print(f"  Download failed.")

    if show_progress:
        print(f"Done processing {reporter}.")


def main():
    parser = argparse.ArgumentParser(
        description="Download reporter volumes from static.case.law"
    )
    parser.add_argument(
        "--reporter", required=True, help="Reporter slug (e.g., cal-rptr-3d)"
    )
    parser.add_argument("--output_dir", default="data", help="Root directory for data")
    parser.add_argument(
        "--max_volumes", type=int, default=0, help="Max volumes to download (0 for all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between downloads (default: 1.0)",
    )
    args = parser.parse_args()

    download_single_reporter(
        args.reporter, args.output_dir, args.max_volumes, args.delay
    )


if __name__ == "__main__":
    main()
