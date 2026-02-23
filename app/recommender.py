from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List

app = FastAPI()

# Dummy book data
BOOKS = [
    {"title": "The Great Gatsby", "description": "A story about the Jazz Age in the United States."},
    {"title": "To Kill a Mockingbird", "description": "A novel about racial injustice in the Deep South."},
    {"title": "1984", "description": "A dystopian novel about totalitarianism and surveillance."},
    {"title": "Pride and Prejudice", "description": "A classic romance novel set in 19th-century England."},
    {"title": "Moby Dick", "description": "A tale of obsession and revenge on the high seas."}
]

# Load the semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')
book_embeddings = model.encode([book["description"] for book in BOOKS], convert_to_tensor=True)

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/recommend")
def recommend_books(request: RecommendationRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, book_embeddings)[0]
    top_results = similarities.topk(request.top_k)
    recommended = []
    for idx in top_results.indices:
        book = BOOKS[idx]
        recommended.append(book)
    return {"recommendations": recommended}
