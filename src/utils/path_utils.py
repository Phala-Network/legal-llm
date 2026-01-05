import os

def normalize_case_path(path: str) -> str:
    """
    Normalizes a path like '/us/213/0301-01' or 'us/1/json/0001-01.json'
    to 'us/1/0001-01'
    """
    if not path:
        return ""

    path = path.lstrip("/")
    if path.endswith(".json"):
        path = path[:-5]

    # Use os.sep for splitting to be cross-platform, though these specific paths likely use /
    parts = path.replace("\\", "/").split("/")

    # Filter out 'json' directory from parts
    parts = [p for p in parts if p and p != "json"]

    return "/".join(parts)
