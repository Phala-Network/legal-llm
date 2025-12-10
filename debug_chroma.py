import chromadb
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("law_cases")
print(f"Collection count: {collection.count()}")
print(f"Peek: {collection.peek()}")
