import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from joblib import load
from tqdm import tqdm

# Imports de vos modules
from model.SolarLSTM import SolarLSTM
from data.solarSystemDataSet import SequentialSolarSystemDataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 15  # Fenêtre d'historique (jours)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# 1. Chargement des données
# Assurez-vous que le chemin pointe vers vos données d'entraînement JSON
df_train = pd.read_json("data/body_coordinates_and_velocities_from_1749-12-31_to_2200-01-09.json", lines=True)  #
# Mettre le bon chemin
scaler = load("model_25-11-25/scaler.joblib")

# 2. Création du Dataset et DataLoader
train_dataset = SequentialSolarSystemDataset(df_train, scaler, sequence_length=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Initialisation du modèle
# On récupère le nombre de corps dynamiquement depuis le dataset
sample_seq, _ = train_dataset[0]
num_bodies = sample_seq.shape[1]  # ex: 9

model = SolarLSTM(num_bodies=num_bodies).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"Modèle LSTM initialisé sur {DEVICE} pour {num_bodies} corps.")

# 4. Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{EPOCHS}")

        for sequences, targets in tepoch:
            # sequences : (Batch, Seq_Len, N_corps, 7)
            # targets   : (Batch, N_corps, 6)
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            predictions = model(sequences)

            # Loss
            loss = criterion(predictions, targets)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_loader):.6f}")

# 5. Sauvegarde
torch.save(model, "model_lstm.pth")
print("Modèle sauvegardé sous model_lstm.pth")