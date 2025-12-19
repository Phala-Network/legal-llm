import sys
import os
import json
import shutil
import tempfile

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag.case_parser import CaseParser


def test_case_parser():
    # Setup temp data dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Structure:
        # tmp_dir/vol_a/json/case_a.json (Cites B)
        # tmp_dir/vol_b/json/case_b.json (Cited by A)

        os.makedirs(os.path.join(tmp_dir, "vol_a", "json"), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, "vol_b", "json"), exist_ok=True)

        case_a_path = os.path.join(tmp_dir, "vol_a", "json", "case_a.json")
        case_b_path = os.path.join(tmp_dir, "vol_b", "json", "case_b.json")

        # Create Case B (The cited case)
        case_b_data = {
            "id": 200,
            "name": "Case B",
            "citation": "200 Cal. 3d 200",
            "casebody": {
                "head_matter": "Summary of Case B.",
                "opinions": [{"text": "Opinion text of Case B."}],
            },
        }
        with open(case_b_path, "w") as f:
            json.dump(case_b_data, f)

        # Create Case A (The citing case)
        # We need to construct the relative path that CaseParser expects.
        # CaseParser.get_local_path logic:
        # input: /cal-rptr-3d/vol/case
        # it splits by /
        # We need to match the structure relative to data_dir

        # If data_dir = tmp_dir
        # And we reference "vol_b/json/case_b"
        # The parser logic:
        # parts = path.split('/')
        # candidate = os.path.join(data_dir, parts[0], parts[1], "json", parts[2] + ".json") (if len parts >=3)
        # That assumes reporter/vol/case format.

        # Let's fake the reporter name as "vol_b" so parts[0]="vol_b".
        # But wait, our directory structure is: tmp_dir/vol_b/json/case_b.json
        # If reporter is parts[0], then path should be "vol_b/anything/case_b"? No.

        # Let's look at parser logic again:
        # candidate = os.path.join(self.data_dir, parts[0], parts[1], "json", f"{parts[2]}.json")
        # So we need 3 parts.
        # Let's make the path: "/vol_b/000/case_b"
        # parts[0]="vol_b", parts[1]="000", parts[2]="case_b"
        # Then file path = tmp_dir/vol_b/000/json/case_b.json

        # Adjusting directory structure to match default parser expectations
        os.makedirs(os.path.join(tmp_dir, "vol_b", "000", "json"), exist_ok=True)
        case_b_path_real = os.path.join(tmp_dir, "vol_b", "000", "json", "case_b.json")
        with open(case_b_path_real, "w") as f:
            json.dump(case_b_data, f)

        # Case A Cites Case B
        case_a_data = {
            "id": 100,
            "name": "Case A",
            "casebody": {
                "head_matter": "Summary of Case A.",
                "opinions": [
                    {"text": "I.\nINTRODUCTION\nThis is Case A opinion text."},
                    {"text": "II.\nDISCUSSION\nWe cite Case B."},
                ],
            },
            "cites_to": [
                {"cite": "200 Cal. 3d 200", "case_paths": ["/vol_b/000/case_b"]}
            ],
        }

        parser = CaseParser(data_dir=tmp_dir)

        print("Testing basic parsing...")
        parsed = parser.parse_case_structure(case_a_data)
        print("Head Matter:", parsed["head_matter"])
        assert "Summary of Case A" in parsed["head_matter"]
        print("Content Blocks:", len(parsed["content_blocks"]))
        # Should have split "I.\nINTRODUCTION..."?
        # My parser implementation just splits.
        # "I.\nINTRODUCTION" -> regex match?
        # Let's see.

        print("Testing recursive retrieval...")
        full_text = parser.extract_full_text(case_a_data, include_recursive=True)
        print("Full Text Preview:\n", full_text)

        assert "Summary of Case A" in full_text
        assert "This is Case A opinion text" in full_text
        assert "CITED MATERIALS" in full_text
        assert "Summary of Case B" in full_text
        assert "Opinion text of Case B" in full_text

        print("\nSUCCESS: All checks passed!")


if __name__ == "__main__":
    test_case_parser()
