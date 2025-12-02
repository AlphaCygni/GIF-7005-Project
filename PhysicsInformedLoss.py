import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    def __init__(self, scaler, num_bodies=9, delta_scaler=1000.0, dt=1.0):
        super().__init__()
        self.num_bodies = num_bodies
        self.delta_scaler = delta_scaler
        self.dt = dt
        
        # Gravitational Constant in Astronomical Units and days
        self.G = 1.48818e-34 

        # We extract the mean and scale from the scaler and register them as 
        # buffers. This allows them to move to GPU automatically with the model.
        # Shape: [1, 7] (x, y, z, vx, vy, vz, mass)
        self.register_buffer('means', torch.tensor(scaler.mean_, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('scales', torch.tensor(scaler.scale_, dtype=torch.float32).unsqueeze(0))

    def forward(self, pred_delta_normalized, input_normalized, batch_vector, reduction='mean'):
        """
        Args:
            pred_delta_normalized: Model output [N_nodes, 6] (scaled delta pos/vel)
            input_normalized: Model input [N_nodes, 7] (pos, vel, mass)
            batch_vector: PyG batch index vector [N_nodes]
            reduction: 'mean' (default), 'sum', or 'none'. 
                       'none' returns a vector of shape [N_nodes], useful for analysis per planet.
        """
        
        # --- 1. Differentiable Inverse Transform to Physical Units ---
        
        # Recover Real Input State (Position T, Velocity T, Mass)
        input_real = input_normalized * self.scales + self.means
        
        pos_t_real = input_real[:, 0:3] # [N, 3]
        mass_log_real = input_real[:, 6].unsqueeze(1) # [N, 1]
        
        # Undo log10 mass
        mass_real = 10 ** mass_log_real 

        # Recover Real Predicted Velocity Delta
        pred_delta_norm = pred_delta_normalized / self.delta_scaler
        
        # Extract only velocity scales (indices 3,4,5)
        vel_scales = self.scales[:, 3:6] 
        
        # Calculate Model Implied Acceleration
        # a = delta_v / dt
        pred_delta_vel_norm = pred_delta_norm[:, 3:6]
        pred_delta_vel_real = pred_delta_vel_norm * vel_scales
        acc_model = pred_delta_vel_real / self.dt

        # --- 2. Calculate Theoretical Newton Acceleration ---
        
        # Batch size handling
        batch_size = batch_vector[-1].item() + 1
        
        # Reshape to [B, N, 3] and [B, N, 1]
        pos_reshaped = pos_t_real.view(batch_size, self.num_bodies, 3) 
        mass_reshaped = mass_real.view(batch_size, self.num_bodies, 1)

        # Pairwise differences and distances
        diff = pos_reshaped.unsqueeze(2) - pos_reshaped.unsqueeze(1) # [B, N, N, 3]
        dist = torch.norm(diff, dim=3, keepdim=True) + 1e-6 # [B, N, N, 1]
        
        # Direction
        direction = diff / dist # [B, N, N, 3]
        
        # Force Magnitude: F/m_i = G * m_j / r^2
        acc_magnitude = self.G * mass_reshaped.unsqueeze(1) / (dist ** 2) # [B, N, N, 1]
        
        # Force Vectors
        acc_vectors = direction * acc_magnitude # [B, N, N, 3]
        
        # Sum forces from all other bodies
        acc_newton_reshaped = torch.sum(acc_vectors, dim=2) # [B, N, 3]
        
        # Flatten back to [N_nodes, 3]
        acc_newton = acc_newton_reshaped.view(-1, 3)

        # --- 3. Compute Loss with Reduction Logic ---
        
        # Calculate squared error for each component (x, y, z)
        # Shape: [N_nodes, 3]
        squared_errors = (acc_model - acc_newton) ** 2

        # Sum the errors of (x,y,z) to get the total squared error magnitude per body
        # Shape: [N_nodes]
        loss_per_node = torch.sum(squared_errors, dim=1)

        if reduction == 'mean':
            return torch.mean(loss_per_node)
        elif reduction == 'sum':
            return torch.sum(loss_per_node)
        elif reduction == 'none':
            return loss_per_node # Returns [N_nodes] for per-planet analysis
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")