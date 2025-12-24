import os
import json
from typing import List, Optional
from datetime import datetime


def get_local_path(case_path: str, data_dir: str) -> Optional[str]:
    """
    Converts a case_path to a local file path.
    Example: '/cal-rptr-3d/211/0149-01' -> '{data_dir}/cal-rptr-3d/211/json/0149-01.json'
    """
    if case_path.startswith("/"):
        case_path = case_path[1:]

    parts = case_path.split("/")
    if len(parts) >= 3:
        # Standard structure: reporter/volume/case
        # We need to handle potential nesting or flat structure if data_dir varies.

        # Check standard: data_dir/reporter/volume/json/case.json
        candidate = os.path.join(
            data_dir, parts[0], parts[1], "json", f"{parts[2]}.json"
        )
        if os.path.exists(candidate):
            return candidate

        # Check nested if data_dir is already the reporter directory
        candidate_nested = os.path.join(data_dir, parts[1], "json", f"{parts[2]}.json")
        if os.path.exists(candidate_nested):
            return candidate_nested

    return None


def filter_cases_by_date(
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_content_length: Optional[int] = None,
) -> List[str]:
    """
    Walks through data_dir to find JSON case files and filters them by decision_date.
    Dates should be in 'YYYY-MM-DD' format.
    If min_content_length is provided, it also filters by the length of opinions text.
    Returns a list of absolute file paths.
    """
    if not os.path.exists(data_dir):
        return []
    matched_files = []

    # Parse dates to objects for comparison
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(".json"):
                continue

            # Skip metadata files
            if "Metadata" in file:
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Date check
                decision_date_str = data.get("decision_date")
                if decision_date_str:
                    try:
                        case_dt = datetime.strptime(decision_date_str, "%Y-%m-%d")
                        if start_dt and case_dt < start_dt:
                            continue
                        if end_dt and case_dt > end_dt:
                            continue
                    except ValueError:
                        continue
                elif start_dt or end_dt:
                    # If date filter active but no date in file, skip
                    continue

                # Content length check
                if min_content_length is not None:
                    opinions = data.get("casebody", {}).get("opinions", [])
                    total_text = "".join([op.get("text", "") for op in opinions])
                    if len(total_text) < min_content_length:
                        continue

                matched_files.append(file_path)

            except (json.JSONDecodeError, OSError):
                continue

    return matched_files


def get_cited_case_paths(case_data: dict, data_dir: str) -> List[str]:
    """
    Extracts local file paths for all cases cited by the given case data.
    """
    cited_paths = []
    citations = case_data.get("cites_to", [])

    for citation in citations:
        paths = citation.get("case_paths", [])
        for p in paths:
            local_p = get_local_path(p, data_dir)
            if local_p and local_p not in cited_paths:
                cited_paths.append(local_p)

    return cited_paths
