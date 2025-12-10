from unsloth import FastLanguageModel
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.retriever import CaseRetriever
import argparse
import re

class LawAssistant:
    def __init__(self, model_path="lora_model", db_path="chroma_db"):
        self.retriever = CaseRetriever(db_path=db_path)
        
        # Load Fine-tuned Model
        # Using 4-bit loading for efficiency as per training
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)

    def chat(self, user_query, state=None):
        # Initial conversation state
        messages = [
            {"role": "user", "content": user_query}
        ]
        
        # Limit turns to prevent infinite loops
        for turn in range(3): 
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
                temperature = 0.5, # Lower temperature for more deterministic tool use
            )
            
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Since decode returns the full conversation, we need to extract just the new part.
            # However, simpler to just grab the text after the last user message end or similar.
            # But the tokenizer might not easily give us just the new tokens' string perfectly with chat template history included.
            # A robust way is to just look at the end of decoded_output.
            # Or use `tokenizer.decode(outputs[0][inputs.shape[1]:], ...)`
            
            new_response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
            print(f"DEBUG: Model generated: {new_response}")

            # 3. Check for tool use
            search_match = re.search(r"<search>(.*?)</search>", new_response, re.DOTALL)
            
            if search_match:
                query = search_match.group(1).strip()
                print(f"DEBUG: Tool Call Detected: Search('{query}')")
                
                # Append assistant's search call to history
                messages.append({"role": "assistant", "content": new_response})
                
                # 4. Execute RAG
                retrieved_docs = self.retriever.retrieve(query, k=3, state=state)
                context_str = ""
                for i, doc in enumerate(retrieved_docs):
                    context_str += f"Case {i+1}: {doc['metadata']['name']} ({doc['metadata']['citation']})\n"
                    context_str += f"Summary/Excerpt: {doc['text'][:500]}...\n\n"
                
                if not context_str:
                    context_str = "No relevant cases found."

                # 5. Append matching tool output (as User message or Tool message)
                # Qwen 2.5 chat template doesn't strict tool roles, usually User with "Context:" is fine or specific observation format
                # Our training data used: {"role": "user", "content": f"Context:\n{...}"}
                messages.append({"role": "user", "content": f"Context:\n{context_str}\n\n"})
                
                # Continue loop to let model generate answer
                continue
            
            else:
                # No tool call, assumed final answer
                return new_response

        return "Max turns reached."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Legal question to ask")
    parser.add_argument("--state", type=str, help="State to filter by", default="California")
    args = parser.parse_args()
    
    assistant = LawAssistant()
    response = assistant.chat(args.query, state=args.state)
    print("\n\n=== Assistant Response ===\n")
    print(response)
