# Solar System Dynamics Prediction (GIF-7005 Project)

## Project Overview

This project focuses on predicting the coordinates and velocities of solar system bodies using deep learning techniques. It compares the performance of Graph Neural Networks (GNNs) and Long Short-Term Memory (LSTM) networks under different configurations, specifically investigating the impact of incorporating physical laws into the training process (Physics-Informed Machine Learning).

The core objective is to evaluate whether adding a physics-informed loss term—derived from Newton's Law of Universal Gravitation—improves the accuracy and generalizability of the models compared to purely data-driven "vanilla" approaches.

## Key Features

*   **Model Comparison**: Comprehensive evaluation of multiple model architectures:
    *   **GNN (Graph Neural Network)**: Trained on the positions and velocities of both planetery system barycenter and planetary  datasets.
    *   **LSTM (Long Short-Term Memory)**: Trained on Planetary datasets.
*   **Physics-Informed Loss**: implementation of a custom loss function (`PhysicsInformedLoss.py`) that penalizes deviations from Newtonian mechanics, ensuring physically consistent predictions.
*   **Dataset Analysis**: Comparison of results using different reference frames (planetery system barycenter vs. Planetary).
*   **Prediction**: Generation of future trajectories for solar system bodies (e.g., for the year 2025).

## Project Structure

*   **`ModelComparizons.ipynb`**: The main notebook for analyzing and comparing the results of different models. It loads predictions, computes metrics, and visualizes trajectories.
*   **`PhysicsInformedLoss.py`**: A PyTorch module implementing the physics-informed loss function. It calculates the theoretical acceleration based on Newton's laws and compares it with the model's implied acceleration.
*   **`Predict2025.ipynb`**: Notebook dedicated to generating predictions for the year 2025.
*   **`GNNTest.ipynb`**: Development and testing notebook for the GNN architecture.
*   **`GetPlanetsPositionsTest.ipynb`**: Utility for testing data extraction and planetary position calculations.
*   **`data/`**: Directory containing the input datasets (coordinates and velocities of celestial bodies).
*   **`results/`**: Directory storing the output predictions from the trained models.
*   **`model/`**: Directory containing the model architectures.
*   **`GNN_trainer.py`**: Training script for the GNN architecture, logs everything in MLflow.


## GNN Architecture

The model architecture follows the **Interaction Network** paradigm, which aligns closely with the intuition of calculating pairwise forces and aggregating them.

1.  **Graph Construction**:
    *   The solar system is represented as a fully connected graph where **Nodes** are celestial bodies (Sun, planets) and **Edges** represent their interactions (gravity).
    *   **Node Features**: Position ($x, y, z$), Velocity ($v_x, v_y, v_z$), and Mass ($m$).

2.  **Message Passing Mechanism**:
    *   **Edge Model (Interaction)**: For every pair of bodies, a neural network (MLP) takes the features of the two bodies and their relative distance as input and computes a comprehensive **Edge Embedding** (interactions).
    *   **Aggregation**: For each body, the model sums up all incoming edge embeddings. This is analogous to summing the interaction force vectors from all other planets impacting this specific body.
    *   **Node Model (Update)**: A second MLP takes the body's current state and the summed interaction effects to compute a new latent representation of the body's future state.

3.  **Deep & Residual Structure**:
    *   **Encoder**: Projects raw physical data (7 inputs) into a 128-dimensional latent space.
    *   **Processor**: 3 stacked Interaction Network layers with **residual connections** refine these embeddings. This allows the model to learn complex, non-linear dependencies over multiple "steps" of reasoning.
    *   **Decoder**: Projects the final latent state back to physical outputs (predicted change in position and velocity).

## Results

*(This section is reserved for manual insertion of pertinent result images and charts.)*

<br>
<br>
<br>
