import pandas as pd
import numpy as np
import pickle

spotify = pd.read_csv("data/spotify_tracks_clean.csv")

with open("models/content_model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/content_model/similarity_matrix.pkl", "rb") as f:
    top_k_indices, top_k_values = pickle.load(f)

def recommend_content(track_name, top_n=10):
    """
    Returns top-n similar tracks for a given track_name using precomputed top-k similarities.
    """
    if track_name not in spotify["track_name"].values:
        print("Track not found.")
        return []

    idx = spotify.index[spotify["track_name"] == track_name][0]

    top_indices = top_k_indices[idx]
    top_scores = top_k_values[idx]

    sorted_idx = np.argsort(top_scores)[::-1][:top_n]

    results = []
    for i in sorted_idx:
        track_idx = top_indices[i]
        results.append({
            "track_name": spotify.loc[track_idx, "track_name"],
            "artist": spotify.loc[track_idx, "artists"],
            "similarity": float(top_scores[i])
        })
    return results

if __name__ == "__main__":
    recommendations = recommend_content("Yellow", top_n=10)
    for r in recommendations:
        print(r)
