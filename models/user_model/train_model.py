import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(MODEL_DIR, "../../data")

lastfm = pd.read_csv(os.path.join(DATA_DIR, "lastfm_clean.csv"))

lastfm["user_idx"] = lastfm["Username"].astype("category").cat.codes
lastfm["track_idx"] = lastfm["Track"].astype("category").cat.codes

n_users = lastfm["user_idx"].nunique()
n_tracks = lastfm["track_idx"].nunique()

user_ids = torch.tensor(lastfm["user_idx"].values, dtype=torch.long)
track_ids = torch.tensor(lastfm["track_idx"].values, dtype=torch.long)

user_mapping = dict(enumerate(lastfm["Username"].astype("category").cat.categories))
with open(os.path.join(MODEL_DIR, "user_mapping.pkl"), "wb") as f:
    pickle.dump(user_mapping, f)

class LastFMDataset(Dataset):
    def __init__(self, users, items):
        self.users = users
        self.items = items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

dataset = LastFMDataset(user_ids, track_ids)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

class CFModel(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

    def forward(self, user, item):
        return (self.user_emb(user) * self.item_emb(item)).sum(1)

model = CFModel(n_users, n_tracks, emb_dim=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
EPOCHS = 5

print("Training Collaborative Filtering model...")
for epoch in range(EPOCHS):
    total_loss = 0
    for u, t in loader:
        pred = model(u, t)
        loss = loss_fn(pred, torch.ones_like(pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
print(f"Model saved as {os.path.join(MODEL_DIR, 'model.pth')}")