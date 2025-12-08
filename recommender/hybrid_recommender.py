import os
import pickle
import torch
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
CONTENT_MODEL_DIR = os.path.join(BASE_DIR, "../models/content_model")
USER_MODEL_DIR = os.path.join(BASE_DIR, "../models/user_model")

spotify = pd.read_csv(os.path.join(DATA_DIR, "spotify_tracks_clean.csv"))
lastfm = pd.read_csv(os.path.join(DATA_DIR, "lastfm_clean.csv"))

with open(os.path.join(CONTENT_MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(CONTENT_MODEL_DIR, "similarity_matrix.pkl"), "rb") as f:
    top_k_indices, top_k_values = pickle.load(f)

class CFModel(torch.nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.item_emb = torch.nn.Embedding(n_items, emb_dim)

    def forward(self, user, item):
        return (self.user_emb(user) * self.item_emb(item)).sum(1)

with open(os.path.join(USER_MODEL_DIR, "user_mapping.pkl"), "rb") as f:
    user_mapping = pickle.load(f)

n_users = len(user_mapping)
n_tracks = len(spotify)

model = CFModel(n_users, n_tracks)
model.load_state_dict(torch.load(os.path.join(USER_MODEL_DIR, "model.pth")))
model.eval()

def recommend_hybrid(track_name, username, top_n=10, w_content=0.7):
    if track_name not in spotify["track_name"].values:
        print("Track not found.")
        return []

    if username not in user_mapping.values():
        print("User not found.")
        return []

    track_idx = spotify.index[spotify["track_name"] == track_name][0]
    content_indices = top_k_indices[track_idx]
    content_scores = top_k_values[track_idx]

    user_idx = list(user_mapping.keys())[list(user_mapping.values()).index(username)]
    user_tensor = torch.tensor([user_idx])
    cf_scores = []
    for item_id in content_indices:
        item_tensor = torch.tensor([item_id])
        score = model(user_tensor, item_tensor).item()
        cf_scores.append(score)
    cf_scores = np.array(cf_scores)

    hybrid_scores = w_content * content_scores + (1 - w_content) * cf_scores

    top_indices_sorted = np.argsort(hybrid_scores)[::-1][:top_n]
    results = []
    for i in top_indices_sorted:
        idx = content_indices[i]
        results.append({
            "track_name": spotify.loc[idx, "track_name"],
            "artist": spotify.loc[idx, "artists"],
            "score": float(hybrid_scores[i])
        })

    return results

if __name__ == "__main__":
    recommendations = recommend_hybrid("Yellow", "isaac", top_n=10)
    for r in recommendations:
        print(r)