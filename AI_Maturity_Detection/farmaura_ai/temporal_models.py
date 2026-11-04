"""Research-only sequence model factories for future repeated-inspection data."""
from __future__ import annotations


def build_trend_model(architecture: str, feature_count: int, hidden_size: int = 64,
                      layer_count: int = 1, output_classes: int = 3):
    """Build a GRU/LSTM classifier for ordered same-sample observations."""
    import torch.nn as nn

    if architecture not in {"gru", "lstm"}:
        raise ValueError("architecture must be 'gru' or 'lstm'.")
    if min(feature_count, hidden_size, layer_count, output_classes) < 1:
        raise ValueError("Model dimensions must be positive.")
    layer_type = nn.GRU if architecture == "gru" else nn.LSTM

    class TrendModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sequence = layer_type(feature_count, hidden_size, num_layers=layer_count,
                                       batch_first=True, dropout=0.0 if layer_count == 1 else 0.1)
            self.head = nn.Linear(hidden_size, output_classes)

        def forward(self, sequence):
            output = self.sequence(sequence)
            states = output[0] if architecture == "lstm" else output
            return self.head(states[:, -1, :])

    return TrendModel()
