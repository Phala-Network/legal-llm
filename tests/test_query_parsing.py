import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_gen.base_generator import BaseGenerator


def test_parsing():
    gen = BaseGenerator()

    # Test cases
    cases = [
        (
            "KEYWORDS: test; TIME: 2020; COURT: Supreme; JURISDICTION: CA",
            {
                "keywords": "test",
                "time": "2020",
                "court": "Supreme",
                "jurisdiction": "CA",
            },
        ),
        (
            "KEYWORDS: test; JURISDICTION: CA",
            {"keywords": "test", "time": None, "court": None, "jurisdiction": "CA"},
        ),
        (
            "Simple query no structure",
            {
                "keywords": "Simple query no structure",
                "time": None,
                "court": None,
                "jurisdiction": None,
            },
        ),
        (
            "TIME: 2021; COURT: High Court",
            {
                "keywords": None,
                "time": "2021",
                "court": "High Court",
                "jurisdiction": None,
            },
        ),
        (
            "KEYWORDS: test; invalid: key",
            {"keywords": "test", "time": None, "court": None, "jurisdiction": None},
        ),
        ("", {"keywords": None, "time": None, "court": None, "jurisdiction": None}),
    ]

    for input_str, expected in cases:
        result = gen._parse_structured_query(input_str)
        assert (
            result == expected
        ), f"Failed for '{input_str}': expected {expected}, got {result}"
        print(f"Passed: '{input_str}' -> {result}")


if __name__ == "__main__":
    try:
        test_parsing()
        print("\nAll parsing tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
