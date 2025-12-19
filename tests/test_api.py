import requests
import json
import time


def test_models():
    print("Testing /v1/models...")
    try:
        response = requests.get("http://localhost:8000/v1/models")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert "data" in response.json()
    except Exception as e:
        print(f"Error testing /v1/models: {e}")


def test_chat_non_streaming():
    print("\nTesting /v1/chat/completions (non-streaming)...")
    payload = {
        "model": "legal-llm-v1",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert "choices" in response.json()
    except Exception as e:
        print(f"Error testing /v1/chat/completions: {e}")


def test_chat_streaming_rag():
    print("\nTesting /v1/chat/completions (streaming with RAG)...")
    payload = {
        "model": "legal-llm-v1",
        "messages": [
            {
                "role": "user",
                "content": "What is the rule for summary judgment in California?",
            }
        ],
        "stream": True,
    }
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions", json=payload, stream=True
        )
        print(f"Status: {response.status_code}")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            print(content, end="", flush=True)
                        if data["choices"][0].get("finish_reason") == "stop":
                            print("\n[Stream Finished]")
                    except json.JSONDecodeError:
                        print(f"\n[Non-JSON data: {data_str}]")
        assert response.status_code == 200
    except Exception as e:
        print(f"Error testing /v1/chat/completions streaming: {e}")


if __name__ == "__main__":
    test_models()
    test_chat_non_streaming()
    test_chat_streaming_rag()
