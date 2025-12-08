import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

TOP_K = 1500  

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(MODEL_DIR, "../../data")
SPOTIFY_FILE = os.path.join(DATA_DIR, "spotify_clean.csv")

spotify = pd.read_csv(SPOTIFY_FILE)

spotify["combined"] = spotify["track_name"] + " " + spotify["artists"] + " " + spotify["track_genre"]

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(spotify["combined"])

nn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine", algorithm="brute")
nn.fit(tfidf_matrix)

distances, indices = nn.kneighbors(tfidf_matrix)

similarity_values = 1 - distances

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "wb") as f:
    pickle.dump((indices, similarity_values), f)

print("Training complete! Vectorizer and top-k similarity matrix saved.")