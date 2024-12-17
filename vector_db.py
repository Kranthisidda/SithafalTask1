import numpy as np
from langchain.vectorstores import FAISS

class VectorDatabase:
    def __init__(self):
        self.embeddings = []
        self.metadata = []
        self.vector_store = FAISS()

    def store_embeddings(self, embeddings, metadata):
        self.embeddings.append(embeddings)
        self.metadata.append(metadata)
        self.vector_store.add(embeddings, metadata)

    def retrieve_similar(self, query_embedding, top_k=5):
        # Simple similarity search using cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.metadata[i] for i in top_indices]