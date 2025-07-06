import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000/recommend"

st.title("🎬 Movie Recommendation System")

movie_df = pd.read_csv("data/ml-latest-small/movies.csv")
movie_dict = dict(zip(movie_df['title'], movie_df['movieId']))

selected_movie = st.selectbox("Choose a movie:", list(movie_dict.keys()))

if st.button("Recommend"):
    movie_id = movie_dict[selected_movie]
    response = requests.get(API_URL, params={"movie_id": movie_id})
    if response.status_code == 200:
        recommendations = response.json()["recommendations"]
        st.subheader("Recommended Movies:")
        for rec in recommendations:
            st.markdown(f"- {rec['title']}")
    else:
        st.error("Error fetching recommendations")