import torch
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "../data")
MODEL_DIR = os.path.join(BASE_DIR, "../models/user_model")

with open(os.path.join(MODEL_DIR, "user_mapping.pkl"), "rb") as f:
    user_mapping = pickle.load(f)

n_users = len(user_mapping)

lastfm = pd.read_csv(os.path.join(DATA_DIR, "lastfm_clean.csv"))
track_mapping = dict(enumerate(lastfm["Track"].astype("category").cat.categories))
n_tracks = len(track_mapping)

class CFModel(torch.nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = torch.nn.Embedding(n_users, emb_dim)
        self.item_emb = torch.nn.Embedding(n_items, emb_dim)

    def forward(self, user, item):
        return (self.user_emb(user) * self.item_emb(item)).sum(1)

model = CFModel(n_users, n_tracks)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pth")))
model.eval()

def recommend_for_user(username, top_n=10):
    if username not in user_mapping.values():
        print("User not found.")
        return []

    user_idx = list(user_mapping.keys())[list(user_mapping.values()).index(username)]
    user_tensor = torch.tensor([user_idx])

    scores = []
    for item_id in range(n_tracks):
        item_tensor = torch.tensor([item_id])
        score = model(user_tensor, item_tensor).item()
        scores.append((item_id, score))

    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for item_id, score in top_scores:
        results.append({
            "track": track_mapping[item_id],
            "score": float(score)
        })

    return results

if __name__ == "__main__":
    recommendations = recommend_for_user("isaac", top_n=10)
    for r in recommendations:
        print(r)