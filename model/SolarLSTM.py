import torch
import torch.nn as nn


class SolarLSTM(nn.Module):
    """
    Modèle de prédiction de trajectoires basé sur LSTM.
    Traite l'état complet du système solaire comme une série temporelle.
    """

    def __init__(self, num_bodies: int, input_features: int = 7, hidden_size: int = 128, num_layers: int = 2,
                 output_features: int = 6):
        super(SolarLSTM, self).__init__()

        self.num_bodies = num_bodies
        self.input_features = input_features

        # L'entrée du LSTM est la concaténation des features de tous les corps
        # Taille = 9 corps * 7 features (x, y, z, vx, vy, vz, m) = 63
        self.flat_input_size = num_bodies * input_features

        self.lstm = nn.LSTM(
            input_size=self.flat_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (Batch, Seq, Features)
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Tête de lecture qui re-projette vers l'espace physique
        # Sortie = 9 corps * 6 features (dx, dy, dz, dvx, dvy, dvz) = 54
        self.flat_output_size = num_bodies * output_features

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.flat_output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tenseur de forme (Batch, Sequence_Length, Num_Bodies, Input_Features)
        Returns:
            Prédiction de forme (Batch, Num_Bodies, Output_Features)
        """
        batch_size, seq_len, _, _ = x.shape

        # 1. Aplatir les dimensions des corps et des features pour le LSTM
        # (Batch, Seq, 9, 7) -> (Batch, Seq, 63)
        x_flat = x.view(batch_size, seq_len, -1)

        # 2. Passage dans le LSTM
        # out contient la séquence des états cachés, on veut juste le dernier
        lstm_out, _ = self.lstm(x_flat)
        last_hidden_state = lstm_out[:, -1, :]  # (Batch, Hidden_Size)

        # 3. Décodage et remise en forme
        prediction_flat = self.decoder(last_hidden_state)

        # (Batch, 54) -> (Batch, 9, 6)
        prediction = prediction_flat.view(batch_size, self.num_bodies, -1)

        return prediction