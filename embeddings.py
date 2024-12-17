from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)