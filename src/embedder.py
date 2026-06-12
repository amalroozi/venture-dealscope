from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(descriptions):
    return model.encode(descriptions, show_progress_bar=True)