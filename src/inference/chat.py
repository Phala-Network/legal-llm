from unsloth import FastLanguageModel
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import CaseRetriever
import argparse
import re
from dotenv import load_dotenv

load_dotenv()

class LawAssistant:
    def __init__(self, model_path="lora_model"):
        # Initialize Retriever (loads Reranker + BM25 + Vector DB)
        self.retriever = CaseRetriever()
        
        # Load Fine-tuned Model
        # If lora_model doesn't exist yet, we fall back to base model for demo, 
        # or error out if strict. Assuming user will run train.py first.
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

    def chat(self, user_query):
        # Initial conversation state with System Prompt to guide behavior
        messages = [
            {"role": "system", "content": "You are a helpful legal assistant. To search for cases, output a search query wrapped in tags, strictly like this: <search>your query terms</search>. Do not use function call syntax like search(...)."},
            {"role": "user", "content": user_query}
        ]
        
        # Limit turns to prevent infinite loops (Agentic Loop)
        max_turns = 3
        for turn in range(max_turns): 
            # 1. Prepare input
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")

            # 2. Generate
            outputs = self.model.generate(
                input_ids = inputs, 
                max_new_tokens = 512, 
                use_cache = True,
                temperature = 0.3, # Low temp for reasoning
            )
            
            # Decode only the new tokens
            # Simple trick: decode all, then split input length? 
            # Or just decode the suffix. fast_chat_template output is a bit complex.
            # Let's decode everything and find the new part by message length or role.
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the last assistant message
            # Qwen uses specific template. 
            # Heuristic: We know the input messages content. 
            # But let's rely on the fact that we append to `messages` list.
            # The *new* content is what we care about.
            # Only reliable way with unsloth/transformers generation loop without streaming:
            input_len = inputs.shape[1]
            new_tokens = outputs[0][input_len:]
            new_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            print(f"DEBUG [Turn {turn}]: {new_response}")

            # 3. Check for tool use
            # Pattern: <search>query</search>
            search_match = re.search(r"<search>(.*?)</search>", new_response, re.DOTALL)
            
            # Fallback: Check for search("query") or search(query) hallucination
            if not search_match:
                fallback_match = re.search(r"search\((.*?)\)", new_response, re.DOTALL)
                if fallback_match:
                    search_match = fallback_match
                    print("DEBUG: Detected non-standard search(...) format, handling gracefully.")
            
            if search_match:
                query = search_match.group(1).strip()
                print(f"--> Search Action: '{query}'")
                
                # Append assistant's "Thought + Search" to history
                messages.append({"role": "assistant", "content": new_response})
                
                # 4. Execute RAG
                # Note: 'state' filter removed from retrieve() for now as we didn't implement robust extraction in chat.py
                # In a real agent, we could have a <filter> tool or extract entities.
                retrieved_docs = self.retriever.retrieve(query, k=3)
                
                context_str = ""
                for i, doc in enumerate(retrieved_docs):
                    context_str += f"[Result {i+1}] {doc['metadata'].get('name', 'Case')}\n"
                    context_str += f"{doc['text'][:800]}...\n\n"
                
                if not context_str:
                    context_str = "No relevant cases found."

                # 5. Output observation
                messages.append({"role": "user", "content": f"Search Results:\n{context_str}\n\n"})
                
                # Continue loop to let model generate answer based on new context
                continue
            
            else:
                # No tool call, assumed final answer
                return new_response

        return "Max turns reached. (The model got stuck in a loop or failed to answer)."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Legal question to ask")
    args = parser.parse_args()
    
    assistant = LawAssistant()
    response = assistant.chat(args.query)
    print("\n\n=== Final Answer ===\n")
    print(response)
