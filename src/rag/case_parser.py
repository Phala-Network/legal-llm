import os
import json
import re
from .case_filtering import get_local_path, get_cited_case_paths


class CaseParser:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Regex for Roman numeral headers (e.g., "I.\n", "IV.\n")
        self.header_pattern = re.compile(r"^[IVXLCDM]+\.\s*$", re.MULTILINE)

    def get_local_path(self, case_path: str) -> str:
        """
        Converts a case_path like '/cal-rptr-3d/211/0149-01' to a local file path.
        """
        return get_local_path(case_path, self.data_dir)

    def parse_case_structure(self, case_data: dict) -> dict:
        """
        Parses the case data to extract:
        - head_matter (Abstract/Summary)
        - opinions (Structured with headers or flat)
        """
        result = {
            "head_matter": "",
            "cites_to": [],
            # List of {"type": "header"|"paragraph", "text": "..."}
            "structured_content": [],
        }

        # 1. Head Matter
        hm = case_data.get("casebody", {}).get("head_matter")
        if hm and hm.strip():
            result["head_matter"] = hm.strip()

        # 2. Cites
        result["cites_to"] = case_data.get("cites_to", [])

        # 3. Opinions
        opinions = case_data.get("casebody", {}).get("opinions", [])

        full_text_blocks = []

        for op in opinions:
            text = op.get("text", "")
            if not text or not text.strip():
                continue

            # Check for structure
            # We want to preserve structure for chunking, but for simple text extraction we might just concatenation.
            # The prompt asked: "opinions contain structural data that we can use to trunct the contents... E.g. ... opinions can be separated by 'I.\nINTRODUCTION'..."

            # Strategy: Split by header pattern, but keep delimiters.
            # re.split with capturing group keeps separate.

            output_blocks = []  # For this opinion

            # We want to split but keep the header attached to the following text?
            # Or just mark them as headers.

            # If we just split, we get: [pre, I., post, II., post]
            parts = re.split(r"(^[IVXLCDM]+\.\s*.*$)", text, flags=re.MULTILINE)

            # Clean up empty parts
            parts = [p for p in parts if p.strip()]

            for p in parts:
                p = p.strip()
                # Add newline to match multiline regex behavior if needed, or just check format
                if self.header_pattern.match(p + "\n"):
                    # It's a header (Wait, re.split captures the whole line if the regex matches the whole line)
                    # My regex was `^[IVXLCDM]+\.\s*$` which is strict.
                    # But some headers have text like "I.\nINTRODUCTION".
                    # The prompt said "I.\nINTRODUCTION". Two lines? Or one?
                    # "I.\nINTRODUCTION" -> The 'I.' is on one line.

                    # Simple approach: Just store the text. The chunker will handle "meaningful" chunks.
                    # But the request is to "use to trunct the contents".
                    # Meaning: Don't cut in the middle of a section if possible, OR
                    # Use the headers as "Chunk Boundaries".

                    pass

                full_text_blocks.append(p)

        result["content_blocks"] = full_text_blocks
        return result

    def extract_full_text(
        self,
        case_data: dict,
        include_recursive: bool = False,
        visited: set = None,
        depth: int = 0,
        max_depth: int = 1,
    ) -> str:
        """
        Extracts the full text representation.
        If include_recursive is True, it attempts to resolve links and append their content up to max_depth.
        """
        if visited is None:
            visited = set()

        if depth > max_depth:
            print(f"Depth {depth} exceeded max depth {max_depth}")
            return ""

        case_id = str(case_data.get("id", ""))
        if case_id in visited:
            return ""
        visited.add(case_id)

        parsed = self.parse_case_structure(case_data)

        parts = []

        # Add Head Matter
        if parsed["head_matter"]:
            parts.append(f"--- HEAD MATTER ---\n{parsed['head_matter']}")

        # Add Content
        if parsed["content_blocks"]:
            parts.append("--- OPINION ---")
            parts.extend(parsed["content_blocks"])

        # Recursive
        if include_recursive and depth < max_depth:
            cited_texts = []
            cited_files = get_cited_case_paths(case_data, self.data_dir)

            for local_p in cited_files:
                try:
                    with open(local_p, "r") as f:
                        sub_data = json.load(f)

                    sub_text = self.extract_full_text(
                        sub_data,
                        include_recursive=True,
                        visited=visited,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                    if sub_text:
                        # Find the citation string for this path if possible, or just use filename
                        # For simplicity in refactor, we just mark it as Cited Case
                        cited_texts.append(
                            f"\n>>>>> CITED CASE START: {local_p} <<<<<\n{sub_text}\n>>>>> CITED CASE END <<<<<\n"
                        )
                except Exception as e:
                    print(f"Error loading cited case {local_p}: {e}")

            if cited_texts:
                parts.append("\n--- CITED MATERIALS ---")
                parts.extend(cited_texts)

        return "\n\n".join(parts)
