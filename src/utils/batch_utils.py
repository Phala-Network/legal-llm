import time
import os


class BatchManager:
    def __init__(self, client):
        self.client = client

    def upload_file(self, file_path: str, purpose: str = "batch") -> str:
        print(f"Uploading {file_path} to OpenAI...")
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose=purpose)
        print(f"File uploaded. ID: {response.id}")
        return response.id

    def create_batch(
        self,
        input_file_id: str,
        description: str = "Batch Job",
        endpoint: str = "/v1/chat/completions",
    ) -> str:
        print(f"Creating Batch for File ID: {input_file_id}...")
        response = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window="24h",
            metadata={"description": description},
        )
        print(f"Batch created. ID: {response.id}")
        return response.id

    def wait_for_batch(self, batch_id: str, poll_interval: int = 60) -> str:
        print(
            f"Waiting for Batch {batch_id} to complete. Polling every {poll_interval}s..."
        )
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "completed":
                return batch.output_file_id
            elif batch.status in ["failed", "cancelled", "expired"]:
                if hasattr(batch, "errors") and batch.errors:
                    print(f"Errors: {batch.errors}")
                raise Exception(f"Batch failed: {batch.status}")
            time.sleep(poll_interval)

    def download_file(self, file_id: str, output_path: str):
        print(f"Downloading File {file_id} to {output_path}...")
        content = self.client.files.content(file_id).read()
        with open(output_path, "wb") as f:
            f.write(content)
        print("Download complete.")
