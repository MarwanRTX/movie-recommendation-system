from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import torch
from torch import nn
import pickle

# Load movies and model artifacts
movies = pd.read_csv("data/ml-latest-small/movies.csv")
movie_ids = movies['movieId'].tolist()
movie_id_to_index = {id_: i for i, id_ in enumerate(movie_ids)}
index_to_movie_id = {i: id_ for id_, i in movie_id_to_index.items()}

# Dummy embeddings (replace with your real trained model output)
num_movies = len(movie_ids)
movie_embeddings = torch.randn(num_movies, 20)  # Replace this with your real model output

app = FastAPI()

# Enable CORS for frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(movie_id: int, top_k: int = 5):
    if movie_id not in movie_id_to_index:
        return {"error": "Invalid movie_id"}

    idx = movie_id_to_index[movie_id]
    target_embedding = movie_embeddings[idx]
    sims = torch.matmul(movie_embeddings, target_embedding)
    top_indices = torch.topk(sims, top_k + 1).indices.tolist()
    top_indices = [i for i in top_indices if i != idx][:top_k]
    
    results = []
    for i in top_indices:
        m_id = index_to_movie_id[i]
        title = movies[movies['movieId'] == m_id]['title'].values[0]
        results.append({"movie_id": m_id, "title": title})
    return {"recommendations": results}

