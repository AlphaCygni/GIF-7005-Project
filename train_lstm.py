import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from joblib import load
from tqdm import tqdm
import numpy as np
from model.SolarLSTM import SolarLSTM
from data.solarSystemDataSet import SequentialSolarSystemDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PhysicsInformedLoss import PhysicsInformedLoss
import joblib

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 15  # Fenêtre d'historique (jours)
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
PHYSICS_LOSS_MULTIPLIER = 1e8 # Multiplicateur pour pondérer la perte associée aux lois physiques
STANDARD_LOSS_WEIGHT = 0.5 # Poids de la fonction de perte standard sur la perte totale

# 1. Chargement des données et split en jeux d'entraînement et de test
# Assurez-vous que le chemin pointe vers vos données d'entraînement JSON
dataframe = pd.read_json("data/body_coordinates_and_velocities_from_1749-12-31_to_2200-01-09.json", lines=True)
timesteps = dataframe['datetime_jd'].unique()
train_timesteps, val_timesteps = train_test_split(timesteps, train_size=0.8, shuffle=False)
df_train = dataframe[dataframe['datetime_jd'].isin(train_timesteps)]
df_train['body_mass'] = np.log10(df_train['body_mass'])
df_val = dataframe[dataframe['datetime_jd'].isin(val_timesteps)]
df_val['body_mass'] = np.log10(df_val['body_mass'])

# Initialisation du scaler à partir des données d'entraînement
feature_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'body_mass']
scaler = StandardScaler()
scaler.fit(df_train[feature_columns].values)

scaler_path = "scaler.joblib"
joblib.dump(scaler, scaler_path)

# 2. Création du Dataset et DataLoader
train_dataset = SequentialSolarSystemDataset(df_train, scaler, sequence_length=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataset = SequentialSolarSystemDataset(df_val, scaler, sequence_length=SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Initialisation du modèle
# On récupère le nombre de corps dynamiquement depuis le dataset
sample_seq, _ = train_dataset[0]
num_bodies = sample_seq.shape[1]  # ex: 9

DELTA_SCALER = 1000.0 # Facteur d'ajustement d'échelle pour les delta de position et de vélocité
model = SolarLSTM(num_bodies=num_bodies).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
std_criterion = nn.MSELoss()
phy_criterion = PhysicsInformedLoss(scaler=scaler, delta_scaler=DELTA_SCALER, dt=1.0, num_bodies=num_bodies).to(DEVICE)

print(f"Modèle LSTM initialisé sur {DEVICE} pour {num_bodies} corps.")

best_val_loss = float('inf')
best_model_path = "best_lstm_model_weights.pth"

# 4. Boucle d'entraînement
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch + 1}/{EPOCHS}")

        for sequences, targets in tepoch:
            # sequences : (Batch, Seq_Len, N_corps, 7)
            # targets   : (Batch, N_corps, 6)
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            predictions = model(sequences)

            # Calcul des valeurs normalisées attendues
            last_inputs_in_sequences = sequences[:, -1, :, :]
            last_inputs_pos_vel = last_inputs_in_sequences[:, :, 0:6]
            target_delta_scaled = targets * DELTA_SCALER

            # Préparation des tenseurs utilisés pour le calcul de la perte physique
            B, N, _ = predictions.shape
            predictions_flatten = predictions.reshape(B * N, 6)
            inputs_flatten  = last_inputs_in_sequences.reshape(B * N, 7)
            batch_vector = torch.arange(B, device=predictions.device, dtype=torch.long).repeat_interleave(N)

            # Loss
            train_loss_std = std_criterion(predictions, target_delta_scaled)
            train_loss_phy = phy_criterion(predictions_flatten, inputs_flatten, batch_vector) * PHYSICS_LOSS_MULTIPLIER
            train_loss = train_loss_std * STANDARD_LOSS_WEIGHT + (1 - STANDARD_LOSS_WEIGHT) * train_loss_phy

            # Backward
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            tepoch.set_postfix(loss=train_loss.item())

    print(f"Epoch {epoch + 1} - Train Avg Loss: {total_train_loss / len(train_loader):.6f}")

    # Phase de validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{EPOCHS}")

            for sequences, targets in tepoch:
                # sequences : (Batch, Seq_Len, N_corps, 7)
                # targets   : (Batch, N_corps, 6)
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)

                # Forward
                predictions = model(sequences)
                
                # Calcul des valeurs normalisées attendues
                last_inputs_in_sequences = sequences[:, -1, :, :]
                last_inputs_pos_vel = last_inputs_in_sequences[:, :, 0:6]
                target_delta_scaled = targets * DELTA_SCALER

                # Préparation des tenseurs utilisés pour le calcul de la perte physique
                B, N, _ = predictions.shape
                predictions_flatten = predictions.reshape(B * N, 6)
                inputs_flatten  = last_inputs_in_sequences.reshape(B * N, 7)
                batch_vector = torch.arange(B, device=predictions.device, dtype=torch.long).repeat_interleave(N)

                # Loss
                val_loss_std = std_criterion(predictions, target_delta_scaled)
                val_loss_phy = phy_criterion(predictions_flatten, inputs_flatten, batch_vector) * PHYSICS_LOSS_MULTIPLIER
                val_loss = val_loss_std * STANDARD_LOSS_WEIGHT + (1 - STANDARD_LOSS_WEIGHT) * val_loss_phy

                # Accumulation
                total_val_loss += val_loss.item()

        # Calcul des Moyennes Finales
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Avg Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model weights locally to disk
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> New Best Model Found! Loss: {best_val_loss:.6f}")

# 5. Sauvegarde
model.load_state_dict(torch.load(best_model_path))
final_model_save_path = "model_lstm_pi.pth"
torch.save(model, final_model_save_path)
print(f"Meilleur modèle sauvegardé sous {final_model_save_path}.")