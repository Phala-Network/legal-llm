import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.ingest import CaseIngester


class TestBatchIngestion(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_batch_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(os.path.join(self.test_dir, "case_1", "json"))
        self.batch_dir = os.path.join(self.test_dir, "batch_temp")

        # Create a dummy case file
        self.case_data = {
            "id": 12345,
            "name": "Test Case",
            "decision_date": "2023-01-01",
            "jurisdiction": {"name_long": "Test Jurisdiction"},
            "citations": [{"cite": "123 Test 456"}],
            "casebody": {
                "head_matter": "Head of the case.",
                "opinions": [
                    {
                        "text": "\n\n".join(
                            [
                                f"Paragraph {i} with some extra text to ensure we hit limits eventually if we want."
                                for i in range(50)
                            ]
                        )
                    }
                ],
            },
        }

        with open(os.path.join(self.test_dir, "case_1", "json", "case.json"), "w") as f:
            json.dump(self.case_data, f)

        # Mock ShardManager
        self.mock_shard_manager = MagicMock()
        self.mock_shard_manager.get_shards_for_case.return_value = [1, 2]
        self.mock_shard_manager.load_assignments.return_value = True

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # ingest_meta.jsonl and ingest_batch_input_1.jsonl are now in batch_dir which is inside test_dir

    @patch("src.rag.ingest.ShardManager")
    @patch("src.rag.ingest.BatchManager")
    @patch("src.rag.ingest.chromadb.PersistentClient")
    @patch("src.rag.ingest.OpenAI")
    def test_submit_batch_ingestion(
        self, MockOpenAI, MockChroma, MockBatchManager, MockShardManager
    ):
        # Setup Mocks
        mock_batch_mgr = MockBatchManager.return_value
        mock_batch_mgr.upload_file.return_value = "file-123"
        mock_batch_mgr.create_batch.return_value = "batch-abc"

        ingester = CaseIngester(
            data_dir=self.test_dir,
            shard_assignments="dummy.json",
            batch_dir=self.batch_dir,
        )
        ingester.shard_manager = (
            self.mock_shard_manager
        )  # Override with our configured mock

        # Run Ingestion
        # Hack to test splitting: we need enough data.
        # Instead of generating 50k items, let's just trust the manual verification or refactor ingest.py to allow limit override.
        # Refactoring ingest.py is better.
        ingester.BATCH_LIMIT = 1
        ingester.ingest(search_dir=self.test_dir, batch_mode=True)

        # Verify Batch Input File created
        batch_input_file = os.path.join(self.batch_dir, "ingest_batch_input_1.jsonl")
        self.assertTrue(os.path.exists(batch_input_file))
        with open(batch_input_file, "r") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            req = json.loads(lines[0])
            self.assertEqual(req["method"], "POST")
            self.assertEqual(req["url"], "/v1/embeddings")

        # Verify Metadata File
        self.assertTrue(
            os.path.exists(os.path.join(self.batch_dir, "ingest_meta.jsonl"))
        )

        # Verify Batch Manager calls
        mock_batch_mgr.upload_file.assert_called()
        mock_batch_mgr.create_batch.assert_called()


if __name__ == "__main__":
    unittest.main()
