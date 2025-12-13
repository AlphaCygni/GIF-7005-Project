# In data/solarSystemDataSet.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from tqdm import tqdm

class SolarSystemDataset(Dataset):
    def __init__(self, dataframe, scaler):
        """
        Args:
            dataframe (pd.DataFrame): The pandas DataFrame for a specific data split (e.g., train or val).
            scaler (sklearn.preprocessing.StandardScaler): The pre-fitted scaler object.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.scaler = scaler

        # --- Data Conversion and Reshaping Logic (Same as before) ---
        self.feature_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'body_mass']
        self.num_features = len(self.feature_columns)

        df_sorted = dataframe.sort_values(['datetime_jd', 'body_id']).reset_index(drop=True)
        grouped = df_sorted.groupby('datetime_jd')

        self.num_timesteps = len(grouped)
        self.num_bodies = len(df_sorted['body_id'].unique())

        self.states = []
        for name, group in tqdm(grouped, total=self.num_timesteps, desc="Processing data groups"):
            if len(group) == self.num_bodies:
                # Get the state as a NumPy array
                state_numpy = group[self.feature_columns].values

                # --- APPLY THE PRE-FITTED SCALER ---
                state_normalized = self.scaler.transform(state_numpy)

                # Convert the *normalized* data to a tensor
                self.states.append(torch.tensor(state_normalized, dtype=torch.float32))

        # --- Edge Index (remains the same) ---
        edge_list = []
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i != j:
                    edge_list.append([i, j])
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def __len__(self):
        return len(self.states) - 1

    def __getitem__(self, idx):
        # The states are already normalized from the __init__ method
        state_t0 = self.states[idx]
        state_t1 = self.states[idx+1]

        graph_sample = Data(x=state_t0, edge_index=self.edge_index)

        # The target 'y' is also normalized. We predict the first 6 features.
        target_state = state_t1[:, 0:6]
        graph_sample.y = target_state

        return graph_sample


class SequentialSolarSystemDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, scaler, sequence_length: int = 10):
        """
        Dataset adaptant les données planétaires en séquences temporelles pour LSTM.
        """
        self.scaler = scaler
        self.sequence_length = sequence_length
        # Colonnes: 7 features (les mêmes que pour le GNN)
        self.feature_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'body_mass']

        # Tri et groupement par date (identique à votre logique existante)
        df_sorted = dataframe.sort_values(['datetime_jd', 'body_id']).reset_index(drop=True)
        grouped = df_sorted.groupby('datetime_jd')

        self.states = []
        # On suppose que chaque groupe contient le même nombre de corps (ex: 9)
        for name, group in tqdm(grouped, desc="Processing sequential data"):
            state_numpy = group[self.feature_columns].values
            # Normalisation
            state_normalized = self.scaler.transform(state_numpy)
            # Stockage sous forme de tenseur (N_corps, 7)
            self.states.append(torch.tensor(state_normalized, dtype=torch.float32))

        if len(self.states) <= sequence_length:
            raise ValueError(f"Pas assez de données ({len(self.states)}) pour la séquence ({sequence_length})")

    def __len__(self):
        # On a besoin de 'sequence_length' jours pour l'entrée + 1 jour pour la cible
        return len(self.states) - self.sequence_length

    def __getitem__(self, idx):
        # Séquence d'entrée (les jours précédents)
        sequence_in = torch.stack(self.states[idx: idx + self.sequence_length])

        # État actuel (dernier jour de la séquence) et état futur (cible)
        state_t = self.states[idx + self.sequence_length - 1]
        state_t1 = self.states[idx + self.sequence_length]

        # On veut prédire la DIFFÉRENCE (le mouvement)
        # Cible = (Position t+1) - (Position t)
        # On ne garde que les 6 premières colonnes (pos, vel)
        target_delta = state_t1[:, 0:6] - state_t[:, 0:6]

        return sequence_in, target_delta