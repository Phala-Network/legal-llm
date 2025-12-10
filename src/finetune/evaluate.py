import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from src.rag.retriever import CaseRetriever
import os
import random

class RAGEvaluator:
    def __init__(self, data_path="training_data.jsonl", num_samples=5):
        self.data_path = data_path
        self.num_samples = num_samples
        self.client = OpenAI()
        self.retriever = CaseRetriever() # Re-uses your hybrid retriever
        
    def load_test_set(self):
        """Loads a hold-out set from the training data."""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # In a real scenario, we'd have a separate test set. 
        # Here we randomly sample to simulate it.
        if len(data) > self.num_samples:
            return random.sample(data, self.num_samples)
        return data

    def extract_qa_pair(self, entry):
        """Extracts Question and Gold Answer from the Multi-turn format."""
        messages = entry.get("messages", [])
        question = ""
        gold_answer = ""
        
        for m in messages:
            if m["role"] == "user" and not question:
                question = m["content"]
            if m["role"] == "assistant":
                # The last assistant message is usually the answer
                gold_answer = m["content"]
                
        return question, gold_answer

    def evaluate(self):
        test_set = self.load_test_set()
        results = []
        
        print(f"Running Evaluation on {len(test_set)} samples...")
        
        for entry in tqdm(test_set):
            question, gold_answer = self.extract_qa_pair(entry)
            if not question: 
                continue
                
            # 1. Run RAG Retrieval
            # Note: We are testing the Retina step primarily here, 
            # as generation depends on the finetuned model which we might not have loaded yet.
            # But we can simulate "System Performance" by seeing if the retrieved context *contains* the answer.
            
            retrieved_docs = self.retriever.retrieve(question, k=5)
            context_text = "\n\n".join([d['text'] for d in retrieved_docs])
            
            # 2. LLM-as-a-Judge
            # We ask GPT-4o to grade if the retrieved context is sufficient.
            
            judge_prompt = f"""
            Task: Evaluate the quality of the Retrieved Context for the given Question.
            
            Question: {question}
            Gold Answer: {gold_answer}
            
            Retrieved Context:
            {context_text[:5000]}
            
            Criteria:
            1. Recall: Does the context contain the information needed to answer the question? (Score 1-5)
            2. Precision: Is the context mostly relevant? (Score 1-5)
            
            Output strictly valid JSON: {{"recall": <int>, "precision": <int>, "reason": "<string>"}}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", # Use mini for judge to save cost, or 4o for accuracy
                    messages=[{"role": "user", "content": judge_prompt}],
                    response_format={ "type": "json_object" }
                )
                score = json.loads(response.choices[0].message.content)
            except Exception as e:
                score = {"recall": 0, "precision": 0, "reason": str(e)}
            
            results.append({
                "question": question,
                "score": score
            })
            
        # Summary
        avg_recall = sum(r['score']['recall'] for r in results) / len(results) if results else 0
        avg_precision = sum(r['score']['precision'] for r in results) / len(results) if results else 0
        
        print(f"\nEvaluation Complete.")
        print(f"Average Context Recall: {avg_recall:.2f}/5")
        print(f"Average Precision: {avg_precision:.2f}/5")
        
        # Save detailed log
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.evaluate()
