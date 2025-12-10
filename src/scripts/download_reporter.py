import os
import json
import requests
import argparse
from tqdm import tqdm
import zipfile
import shutil

METADATA_URL_TEMPLATE = "https://static.case.law/{reporter}/VolumesMetadata.json"
VOLUME_URL_TEMPLATE = "https://static.case.law/{reporter}/{volume_number}.zip"

def download_file(url, output_path):
    """Downloads a file with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def unzip_file(zip_path, extract_to):
    """Unzips a file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error unzipping {zip_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download reporter volumes from static.case.law")
    parser.add_argument("--reporter", required=True, help="Reporter slug (e.g., cal-rptr-3d)")
    parser.add_argument("--output_dir", default="data", help="Root directory for data")
    parser.add_argument("--max_volumes", type=int, default=0, help="Max volumes to download (0 for all)")
    args = parser.parse_args()

    reporter_dir = os.path.join(args.output_dir, args.reporter)
    os.makedirs(reporter_dir, exist_ok=True)
    
    print(f"Setting up for {args.reporter} in {reporter_dir}")

    # 1. Download Metadata
    metadata_url = METADATA_URL_TEMPLATE.format(reporter=args.reporter)
    metadata_path = os.path.join(reporter_dir, "VolumesMetadata.json")
    
    print("Downloading metadata...")
    if not download_file(metadata_url, metadata_path):
        print("Failed to download metadata. Exiting.")
        return

    # 2. Parse Metadata
    try:
        with open(metadata_path, 'r') as f:
            volumes = json.load(f)
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return

    print(f"Found {len(volumes)} volumes.")
    
    # 3. Download Volumes
    count = 0
    for vol in volumes:
        if args.max_volumes > 0 and count >= args.max_volumes:
            print(f"Reached max volumes limit ({args.max_volumes}). Stopping.")
            break

        vol_num = vol.get("volume_number")
        if not vol_num:
            continue
            
        print(f"Processing Volume {vol_num}...")
        
        # Define paths
        zip_name = f"{vol_num}.zip"
        zip_path = os.path.join(reporter_dir, zip_name)
        extract_dir = os.path.join(reporter_dir, vol_num)
        
        # Check if already exists (simple check: if directory exists and not empty)
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"  Volume {vol_num} already appears to be downloaded. Skipping.")
            count += 1
            continue

        # Download
        url = VOLUME_URL_TEMPLATE.format(reporter=args.reporter, volume_number=vol_num)
        if download_file(url, zip_path):
            # Unzip
            print(f"  Unzipping to {extract_dir}...")
            os.makedirs(extract_dir, exist_ok=True)
            if unzip_file(zip_path, extract_dir):
                print(f"  Unzip successful.")
                # Cleanup Zip
                os.remove(zip_path)
                count += 1
            else:
                 print(f"  Unzip failed.")
        else:
            print(f"  Download failed.")

    print("Done.")

if __name__ == "__main__":
    main()
