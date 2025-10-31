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
        
        df_sorted = dataframe.sort_values('datetime_jd').reset_index(drop=True)
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