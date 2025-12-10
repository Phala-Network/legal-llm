import os
import json
import random
import glob
from openai import OpenAI
from tqdm import tqdm
import re
from dotenv import load_dotenv

load_dotenv()

# Initialize client (expects OPENAI_API_KEY env var)
client = OpenAI()

def get_html_text(json_path):
    html_path = json_path.replace('/json/', '/html/').replace('.json', '.html')
    if os.path.exists(html_path):
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple cleanup
                clean = re.compile('<.*?>')
                text = re.sub(clean, ' ', content)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except:
            return ""
    return ""

def generate_synthetic_data(data_dir="data", output_file="training_data.jsonl", num_samples=10):
    json_files = glob.glob(os.path.join(data_dir, "**", "json", "*.json"), recursive=True)
    
    if not json_files:
        print("No data files found.")
        return

    # Sample random cases if we have enough, else take all
    selected_files = random.sample(json_files, min(num_samples, len(json_files)))
    
    with open(output_file, 'w') as out_f:
        for json_file in tqdm(selected_files):
            text = get_html_text(json_file)
            if not text or len(text) < 500:
                continue
                
            # Truncate text to avoid context limits if necessary, though 4o handles 128k
            # But for cost and speed, let's limit context
            context_text = text[:8000] 
            
            prompt = f"""
            You are a legal expert helper. I will provide you with the text of a court case.
            Your task is to generate 3 high-quality training examples based strictly on this case.
            
            For each example, provide:
            1. "question": A challenging question a user might ask.
            2. "query": A search query the assistant would generate to find relevant case law (formatted as keywords).
            3. "relevant_context": A specific excerpt from the text that contains the answer.
            4. "answer": The answer to the question, deriving strictly from the provided context.

            Format the output as a valid JSON list of objects:
            [
                {{"question": "...", "query": "...", "relevant_context": "...", "answer": "..."}},
                ...
            ]
            
            Case Text:
            {context_text}
            """
            
            try:
                print(f"Generating for {json_file}...")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant assisting in creating fine-tuning data for a legal LLM."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }
                )
                
                content = response.choices[0].message.content
                # Sometimes legacy models or certain prompts might not return pure JSON despite instruction, 
                # but "json_object" mode usually enforces it if "JSON" is in prompt.
                # However, the mode returns a JSON object, I asked for a list. 
                # It might wrap it in a key. Let's handle parsing.
                
                try:
                    data = json.loads(content)
                    
                    pairs = []
                    if isinstance(data, list):
                        pairs = data
                    elif isinstance(data, dict):
                        # look for any list value
                        for k, v in data.items():
                            if isinstance(v, list):
                                pairs = v
                                break
                    
                    print(f"Parsed pairs: {pairs}")
                    for pair in pairs:
                        print(f"Keys found: {pair.keys()}")
                        if "question" in pair and "answer" in pair and "query" in pair and "relevant_context" in pair:
                            # Create Multi-turn format for Agentic RAG
                            entry = {
                                "messages": [
                                    {"role": "user", "content": pair["question"]},
                                    {"role": "assistant", "content": f"<search>{pair['query']}</search>"},
                                    {"role": "user", "content": f"Context:\n{pair['relevant_context']}\n\n"},
                                    {"role": "assistant", "content": pair["answer"]}
                                ]
                            }
                            out_f.write(json.dumps(entry) + "\n")
                            
                except json.JSONDecodeError:
                    print(f"Failed to decode JSON for {json_file}")

            except Exception as e:
                print(f"Error generating for {json_file}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    generate_synthetic_data(num_samples=args.num_samples)
