import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class CaseRetriever:
    def __init__(self, db_path="chroma_db", collection_name="law_cases"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def retrieve(self, query, k=5, state=None):
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        where_filter = {}
        if state:
            where_filter["state"] = state
            
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter if where_filter else None
        )
        
        # Flatten results (Chroma returns list of lists)
        retrieved_docs = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = results['distances'][0][i] if results['distances'] else 0
                retrieved_docs.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
        
        return retrieved_docs

if __name__ == "__main__":
    # Simple test
    retriever = CaseRetriever()
    print("Testing retrieval for 'contract breach'...")
    results = retriever.retrieve("contract breach regarding water supply")
    for r in results:
        print(f"--- Score: {r['distance']} ---")
        print(f"Case: {r['metadata']['name']} ({r['metadata']['citation']})")
        print(f"Text snippet: {r['text'][:200]}...")
