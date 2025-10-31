import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# This is the core GNN layer that performs one round of message passing.
class InteractionNetwork(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add') # "add" aggregation.
        
        # Edge Model: computes messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, edge_dim), # Takes concatenated features of two nodes
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Node Model: updates node states
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim), # Takes original node feature + aggregated message
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, x, edge_index):
        # The 'propagate' method calls message(), aggregate(), and update() internally.
        # j: source node, i: target node
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: features of target nodes [num_edges, node_dim]
        # x_j: features of source nodes [num_edges, node_dim]
        
        # Concatenate features of source and target nodes to create edge features
        edge_features = torch.cat([x_i, x_j], dim=1)
        
        # Compute the message using the edge model
        message = self.edge_mlp(edge_features)
        return message

    def update(self, aggregated_messages, x):
        # aggregated_messages: [num_nodes, edge_dim]
        # x: original node features [num_nodes, node_dim]
        
        # Concatenate node features and aggregated messages
        update_input = torch.cat([x, aggregated_messages], dim=1)
        
        # Compute the final updated node representation
        return self.node_mlp(update_input)

# This is the full model that orchestrates the GNN.
class GNN_NBody(nn.Module):
    def __init__(self, input_dim=7, model_dim=128, num_layers=3):
        super().__init__()

        # 1. Encode input features into a higher-dimensional space
        self.node_encoder = nn.Linear(input_dim, model_dim)

        # 2. Stack multiple interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(InteractionNetwork(model_dim, model_dim))

        # 3. Decode the final node embeddings to get predicted acceleration
        self.output_decoder = nn.Linear(model_dim, 6) # Predicts 3D acceleration (x, y, z, vx, vy, vz)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Encode node features
        x = self.node_encoder(x)

        # Pass through interaction layers
        for layer in self.interaction_layers:
            # We add a residual connection for better training stability
            x = x + layer(x, edge_index)

        # Decode to get acceleration
        acceleration = self.output_decoder(x)
        return acceleration
    
    def train_one_epoch(self, training_loader, loss_fn, optimizer):
        running_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            running_loss += loss.item()

            # Adjust learning weights
            optimizer.step()

        return running_loss / len(training_loader)