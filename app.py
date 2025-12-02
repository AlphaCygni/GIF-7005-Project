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
from PhysicsInformedLoss import PhysicsInformedLoss

# --- Start of MLflow Run ---
mlflow.set_experiment("Solar System N-Body GNN Debugging Experiment")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 5. Model ---
    model = GNN_NBody(input_dim=7, model_dim=128).to(device) # Assuming model input is 7 features
    loss_fn = torch.nn.MSELoss()
    loss_fn_physic = PhysicsInformedLoss(scaler=scaler, delta_scaler=params["delta_scaler"], dt=1.0, num_bodies=NUM_BODIES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    # --- 6. Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])

    # --- 7. Full Training and Validation Loop ---
    for epoch in range(params["epochs"]):
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        total_phy_loss = 0
        total_std_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # The model predicts the *normalized* next state (pos, vel)
            pred_delta_normalized = model(batch)
            
            # Compute labels
            input_pos_vel = batch.x[:, 0:6]
            target_pos_vel = batch.y
            raw_delta = target_pos_vel - input_pos_vel
            target_delta_scaled = raw_delta * params["delta_scaler"]

            # Loss is calculated on the normalized values
            loss_std = loss_fn(pred_delta_normalized, target_delta_scaled)
            
            loss_physics = loss_fn_physic(pred_delta_normalized, batch.x, batch.batch)*params["physics_weight"]
            
            loss = loss_std * params["loss_fn_ratio"] + (1 - params["loss_fn_ratio"]) * loss_physics
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            total_std_loss += loss_std.item()
            total_phy_loss += loss_physics.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_std_loss = total_std_loss / len(train_loader)
        avg_phy_loss = total_phy_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        total_val_phy_loss = 0
        total_val_std_loss = 0
        total_val_phy_earth = 0
        best_val_loss = float('inf')
        best_model_path = "best_model_weights.pth"
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_delta_normalized = model(batch)
                
                # --- A. Calcul de la Loss Standard ---
                input_pos_vel = batch.x[:, 0:6]
                target_pos_vel = batch.y
                raw_delta = target_pos_vel - input_pos_vel
                target_delta_scaled = raw_delta * params["delta_scaler"]

                val_loss_std = loss_fn(pred_delta_normalized, target_delta_scaled)

                # --- B. Calcul de la Loss Physique ---
                # On récupère le vecteur d'erreur brut pour chaque corps
                raw_phy_loss_vector = loss_fn_physic(
                    pred_delta_normalized, 
                    batch.x, 
                    batch.batch, 
                    reduction='none' 
                )
                
                # Moyenne GLOBALE (pour guider le modèle)
                val_loss_phy_global = raw_phy_loss_vector.mean() * params["physics_weight"]

                # --- C. Extraction Chirurgicale : TERRE (Index 3) ---
                indices = torch.arange(raw_phy_loss_vector.size(0), device=device)
                
                # Masque pour ne garder que la Terre (Soleil=0, Mercure=1, Venus=2, Terre=3...)
                mask_earth = (indices % NUM_BODIES == 3)
                
                # Moyenne spécifique à la Terre
                val_loss_earth = raw_phy_loss_vector[mask_earth].mean() * params["physics_weight"]

                # --- D. Combinaison et Accumulation ---
                
                # Loss totale (utilise la moyenne globale des 9 corps)
                val_loss = val_loss_std * params["loss_fn_ratio"] + (1 - params["loss_fn_ratio"]) * val_loss_phy_global

                # Accumulation
                total_val_loss += val_loss.item()
                total_val_std_loss += val_loss_std.item()
                total_val_phy_loss += val_loss_phy_global.item()
                
                # Accumulation Terre
                total_val_phy_earth += val_loss_earth.item()

        # --- Calcul des Moyennes Finales ---
        num_batches = len(val_loader)
        
        avg_val_loss = total_val_loss / num_batches
        avg_val_std_loss = total_val_std_loss / num_batches
        avg_val_phy_loss = total_val_phy_loss / num_batches
        
        avg_val_phy_earth = total_val_phy_earth / num_batches

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model weights locally to disk
            torch.save(model.state_dict(), best_model_path)
            print(f"   >>> New Best Model Found! Loss: {best_val_loss:.6f}")

        # Step du scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # --- Logging MLflow ---
        mlflow.log_metric("train_loss_total", avg_train_loss, step=epoch)
        mlflow.log_metric("train_loss_standard", avg_std_loss, step=epoch)
        mlflow.log_metric("train_loss_physics", avg_phy_loss, step=epoch)
        
        mlflow.log_metric("val_loss_total", avg_val_loss, step=epoch)
        mlflow.log_metric("val_loss_standard", avg_val_std_loss, step=epoch)
        mlflow.log_metric("val_loss_physics", avg_val_phy_loss, step=epoch)
        
        # LA métrique importante
        mlflow.log_metric("val_phy_EARTH", avg_val_phy_earth, step=epoch)

        print(f"Epoch {epoch+1:02d} | Train: {avg_train_loss:.5f} (Std: {avg_std_loss:.5f}, Phy: {avg_phy_loss:.5f}) | Val: {avg_val_loss:.5f} (Val Std: {avg_val_std_loss:.5f}, Val Phy: {avg_val_phy_loss:.5f})")
        print(f"   >>> [EARTH VAL PHYSICS LOSS]: {avg_val_phy_earth:.2f}")

    # --- 8. Log the Final Model ---
    print("Logging final model to MLflow...")
    model.load_state_dict(torch.load(best_model_path))
    mlflow.pytorch.log_model(model, "model", registered_model_name="GNN_NBody_Simulator")
    print("\n--- Training Complete ---")
