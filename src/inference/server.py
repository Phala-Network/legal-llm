import os
import glob
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sse_starlette.sse import EventSourceResponse
import asyncio

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import existing logic
from src.rag.retriever import CaseRetriever
from unsloth import FastLanguageModel
import torch
import re

app = FastAPI(title="Legal LLM Server")

# --- Global State ---
CASE_ID_MAP = {} # id -> file_path
law_assistant = None

class LawAssistantWrapper:
    def __init__(self, model_path="lora_model"):
        # Auto-detect collection
        import chromadb
        temp_client = chromadb.PersistentClient(path="chroma_db")
        cols = [c.name for c in temp_client.list_collections() if c.name.startswith("law_cases")]
        target_col = cols[0] if cols else "law_cases"
        print(f"Using collection: {target_col}")

        self.retriever = CaseRetriever(collection_name=target_col)
        
        try:
            print(f"Loading model from {model_path}...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_path,
                max_seq_length = 2048,
                load_in_4bit = True,
            )
            FastLanguageModel.for_inference(self.model)
        except Exception as e:
            print(f"Error loading LoRA model: {e}")
            print("Falling back to base model unsloth/Qwen2.5-7B-Instruct-bnb-4bit...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
                max_seq_length = 2048,
                load_in_4bit = True,
            )
            FastLanguageModel.for_inference(self.model)

    def generate_response(self, messages: List[Dict]):
        # Simple non-streaming implementation for now, or streaming if requested
        # For Open-WebUI, valid OpenAI format is crucial.
        
        # 1. First Pass (Think/Search?)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids = inputs, 
            max_new_tokens = 512, 
            use_cache = True,
            temperature = 0.3,
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        # Check for <search>
        search_match = re.search(r"<search>(.*?)</search>", response_text, re.DOTALL)
        if search_match:
            query = search_match.group(1).strip()
            print(f"Server Search: {query}")
            
            # Retrieve
            retrieved_docs = self.retriever.retrieve(query, k=3)
            context_str = ""
            for i, doc in enumerate(retrieved_docs):
                # We need to make sure the model cites these.
                # Inject mapped ID if possible, though retriever might return internal ID.
                # The retriever returns 'id' which IS the case ID (from ingest logic).
                # But ingest logic made ID = case_id_chunkIndex.
                # We need the base case_id.
                
                # Check ingest.py: doc_id = f"{case_id}_{i}"
                real_case_id = doc["id"].rsplit('_', 1)[0]
                case_name = doc['metadata'].get('name', 'Case')
                
                # Format for context
                context_str += f"[Result {i+1}] {case_name} (ID: {real_case_id})\n"
                context_str += f"{doc['text'][:800]}...\n\n"

            if not context_str:
                context_str = "No relevant cases found."

            # Append to history
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Search Results:\n{context_str}\n\n"})

            # Second Pass
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")

            outputs = self.model.generate(
                input_ids = inputs, 
                max_new_tokens = 1024, 
                use_cache = True,
                temperature = 0.3,
            )
            final_response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            return final_response
        
        return response_text

# --- API Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    stream: bool = False

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    global CASE_ID_MAP, law_assistant
    
    print("Building Case ID Map...")
    data_dir = "data" # Adjust if needed
    json_files = glob.glob(os.path.join(data_dir, "**", "json", "*.json"), recursive=True)
    
    count = 0
    for fpath in json_files:
        try:
            # We need to peek at the ID without full load if possible, but full load is safer
            # Optimized: Just read until ID is found? JSON is tricky.
            # Let's simple load. If 100k files, this is slow. 
            # User has small dataset for now? 
            # Let's assume acceptable for now.
            with open(fpath, 'r', encoding='utf-8') as f:
                # Fast scan: id is usually near top?
                # Case files can be large. reading whole file is bad if just for ID.
                # Using ijson or similar is better, but not installed.
                # Let's read first 2KB and try regex if standard format?
                # Or just load.
                data = json.load(f)
                cid = str(data.get('id'))
                if cid:
                    CASE_ID_MAP[cid] = fpath
                    count += 1
        except Exception as e:
            pass
            
    print(f"Mapped {count} cases.")
    
    # Load Model
    law_assistant = LawAssistantWrapper()

# --- Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Convert Pydantic to dict
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    
    response_text = law_assistant.generate_response(msgs)
    
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

@app.get("/cases/{case_id}")
async def get_case_content(case_id: str):
    if case_id not in CASE_ID_MAP:
        raise HTTPException(status_code=404, detail="Case not found")
    
    fpath = CASE_ID_MAP[case_id]
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format HTML
        name = data.get('name_abbreviation', data.get('name', 'Case'))
        date = data.get('decision_date', 'Unknown Date')
        
        opinions = data.get('casebody', {}).get('opinions', [])
        text_html = ""
        for op in opinions:
            op_text = op.get('text', '').replace('\n', '<br>')
            text_html += f"<h3>{op.get('type', 'Opinion')}</h3><p>{op_text}</p><hr>"
            
        full_html = f"""
        <html>
        <head><title>{name}</title></head>
        <body style="font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px;">
            <h1>{name}</h1>
            <p><strong>Date:</strong> {date}</p>
            <p><strong>ID:</strong> {case_id}</p>
            <hr>
            {text_html}
        </body>
        </html>
        """
        return HTMLResponse(content=full_html)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
