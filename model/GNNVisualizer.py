import torch
from torchview import draw_graph
from GNN import GNN_NBody
from types import SimpleNamespace
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import numpy as np
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.solarSystemDataSet import SolarSystemDataset


model = GNN_NBody(num_layers=1)

class GNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch=None):
        # We reconstruct a fake Data object using SimpleNamespace.
        # This allows your model to call data.x and data.edge_index successfully.
        data = SimpleNamespace(x=x, edge_index=edge_index, batch=batch)
        return self.model(data)

# 1. Load Data
dataframe = pd.read_json("data/body_coordinates_and_velocities_from_1749-12-31_to_2200-01-09.json", lines=True)
dataframe['body_mass'] = np.log10(dataframe['body_mass'])  # Log-transform masses to handle scale

# 2. Split DataFrame chronologically to avoid data leakage
NUM_BODIES = len(dataframe['body_id'].unique())
timesteps = dataframe['datetime_jd'].unique()
train_timesteps, val_timesteps = train_test_split(timesteps, train_size=0.8, shuffle=False)
train_df = dataframe[dataframe['datetime_jd'].isin(train_timesteps)]
val_df = dataframe[dataframe['datetime_jd'].isin(val_timesteps)]

# 3. Fit Scaler on Training Data
feature_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'body_mass']
scaler = StandardScaler()
scaler.fit(train_df[feature_columns].values)

# --- Hyperparameters ---
params = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50, # A reasonable number of epochs for a complex task
    "num_workers": 0, # Keep to 0 on windows or it breaks stuff
    "loss_fn_ratio": 0.5, # Weight for the physics-informed loss
    "delta_scaler": 1000.0, # Scaling factor for delta position/velocity
    "physics_weight": 1e8, # Weight for the physics-informed loss
}

# 4. Instantiate Datasets with the fitted scaler
print("Initializing datasets...")
train_dataset = SolarSystemDataset(train_df, scaler)

# 2. GET ONE SAMPLE FROM YOUR DATASET
# Assuming 'dataset' is already instantiated from your SolarSystemDataset class
sample_graph = train_dataset[0]  # Get the first item (t=0)

# 3. GENERATE THE VISUALIZATION
# We pass the Wrapper instead of the model.
# We pass the tensors (x, edge_index) directly to 'input_data'.
graph = draw_graph(
    GNNWrapper(model),
    input_data=(sample_graph.x, sample_graph.edge_index),
    expand_nested=True,
    depth=4,
    graph_name="SolarSystemGNN3",
    save_graph=True
)

# 2. Change the font to Arial (standard on Windows)
graph.visual_graph.graph_attr['fontname'] = 'Arial'
graph.visual_graph.node_attr['fontname'] = 'Arial'
graph.visual_graph.edge_attr['fontname'] = 'Arial'

# 3. Now render/save the file
graph.visual_graph.render("SolarSystemGNN_2Bodies", format="png", cleanup=True)