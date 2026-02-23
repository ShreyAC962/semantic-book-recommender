# Semantic Book Recommender API

This project is a FastAPI-based semantic book recommender. It uses sentence-transformers to provide book recommendations based on semantic similarity.

## Features
- REST API with FastAPI
- Semantic search using sentence-transformers
- Example endpoint for book recommendations

## Setup

1. **Create and activate a virtual environment** (already done):
   
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies** (already done):
   
   ```sh
   pip install fastapi uvicorn[standard] sentence-transformers
   ```

3. **Run the API server:**
   
   ```sh
   uvicorn app.main:app --reload
   ```

4. **Try the recommender endpoint:**
   
   Send a POST request to `/recommend` with a JSON body:
   ```json
   {
     "query": "dystopian future",
     "top_k": 2
   }
   ```

## Project Structure
- `app/main.py`: FastAPI app entry point
- `app/recommender.py`: Semantic recommendation logic

## Requirements
- Python 3.8+

---

Replace the dummy book data in `recommender.py` with your own dataset for production use.
