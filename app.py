import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Assuming your custom classes are in these locations
from data.solarSystemDataSet import SolarSystemDataset
from model.GNN import GNN_NBody

# --- Start of MLflow Run ---
mlflow.set_experiment("Solar System N-Body GNN Debugging Experiment")

# 1. Load Data
dataframe = pd.read_json("data/body_coordinates_and_velocities_from_1749-12-31_to_2200-01-09.json", lines=True)
dataframe['body_mass'] = np.log10(dataframe['body_mass'])  # Log-transform masses to handle scale

# 2. Split DataFrame chronologically to avoid data leakage
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
}

# 4. Instantiate Datasets with the fitted scaler
print("Initializing datasets...")
train_dataset = SolarSystemDataset(train_df, scaler)
val_dataset = SolarSystemDataset(val_df, scaler)

# Sanity check
sample = train_dataset[0]
pos_t0 = sample.x[:, 0:3]
pos_t1 = sample.x[:, 0:3] + sample.y[:, 0:3] # Reconstruct next step

# If aligned, this difference should be tiny (e.g., 0.0001)
# If misaligned, this will be large (e.g., 1.5)
diff = torch.norm(sample.y[:, 0:3] - sample.x[:, 0:3])
print(f"Average movement per step (Normalized): {diff.item()}")

with mlflow.start_run() as run:
    print(f"Starting MLflow Run: {run.info.run_id}")
    mlflow.log_params(params)

    # Save the scaler as an artifact with the run
    scaler_path = "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path, "scaler")
    print("Scaler saved as MLflow artifact.")

    # --- 5. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 6. Model ---
    model = GNN_NBody(input_dim=7, model_dim=128).to(device) # Assuming model input is 7 features
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    DELTA_SCALER = 1000.0

    # --- 7. Full Training and Validation Loop ---
    for epoch in range(params["epochs"]):
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # The model predicts the *normalized* next state (pos, vel)
            pred_delta_normalized = model(batch)
            
            # Compute labels
            input_pos_vel = batch.x[:, 0:6]
            target_pos_vel = batch.y
            raw_delta = target_pos_vel - input_pos_vel
            target_delta_scaled = raw_delta * DELTA_SCALER

            # Loss is calculated on the normalized values
            loss = loss_fn(pred_delta_normalized, target_delta_scaled)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        total_physical_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_delta_normalized = model(batch)
                
                # Compute labels
                input_pos_vel = batch.x[:, 0:6]
                target_pos_vel = batch.y
                raw_delta = target_pos_vel - input_pos_vel
                target_delta_scaled = raw_delta * DELTA_SCALER

                # 1. Calculate the standard validation loss on normalized data
                val_loss = loss_fn(pred_delta_normalized, target_delta_scaled)
                total_val_loss += val_loss.item()
                
                ## --- 2. Inverse Transform for Physical Error Calculation ---
                ## Move prediction and target to CPU and convert to NumPy
                #pred_norm_np = predicted_normalized.cpu().numpy()
                #target_norm_np = batch.y.cpu().numpy()
#
                ## The scaler expects 7 features, but our prediction/target has 6.
                ## We need to add a dummy column for 'body_mass' to match the scaler's shape.
                ## A column of zeros is fine for this purpose.
                #dummy_mass_col = np.zeros((pred_norm_np.shape[0], 1))
#                
                ## Create full 7-feature arrays
                #pred_full_features = np.hstack([pred_norm_np, dummy_mass_col])
                #target_full_features = np.hstack([target_norm_np, dummy_mass_col])
#
                ## Apply the inverse transformation
                #pred_physical = scaler.inverse_transform(pred_full_features)
                #target_physical = scaler.inverse_transform(target_full_features)
#                
                ## We only care about the first 6 features (pos, vel) for the error calculation
                #pred_physical_pos_vel = pred_physical[:, 0:6]
                #target_physical_pos_vel = target_physical[:, 0:6]
#                
                ## Calculate the Mean Squared Error on the un-normalized, physical values
                #batch_physical_mse = np.mean((pred_physical_pos_vel - target_physical_pos_vel)**2)
                #total_physical_mse += batch_physical_mse

        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        #avg_physical_mse = total_physical_mse / len(val_loader)

        # --- 7. Logging to MLflow and Console ---
        mlflow.log_metric("train_loss_normalized", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss_normalized", avg_val_loss, step=epoch)
        #mlflow.log_metric("val_mse_physical", avg_physical_mse, step=epoch)
        
        # Taking the square root gives an error in the original units (e.g., AU and AU/day)
        #avg_physical_rmse = np.sqrt(avg_physical_mse)
        #mlflow.log_metric("val_rmse_physical", avg_physical_rmse, step=epoch)

        print(f"Epoch {epoch+1:02d}/{params['epochs']} | "
            f"Train Loss (Norm): {avg_train_loss:.6f} | "
            f"Val Loss (Norm): {avg_val_loss:.6f} | ")
            #f"Val RMSE (Physical): {avg_physical_rmse:.6f}")

    # --- 8. Log the Final Model ---
    print("Logging final model to MLflow...")
    mlflow.pytorch.log_model(model, "model", registered_model_name="GNN_NBody_Simulator")
    print("\n--- Training Complete ---")
