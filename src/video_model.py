"""
Model architecture for body pose + hand sign language detection.

Features per frame:
- Body pose: 33 landmarks * 3 (x, y, z) = 99 values
- Left hand: 21 landmarks * 3 (x, y, z) = 63 values
- Right hand: 21 landmarks * 3 (x, y, z) = 63 values
Total: 225 features per frame
"""

import torch
import torch.nn as nn


class BodySignLSTM(nn.Module):
    """
    LSTM model for sign language detection using body pose + hand landmarks.
    """

    def __init__(self, input_size=225, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_type = "lstm"

        # 2-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Classification head
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Use the final time step output
        last_step = lstm_out[:, -1, :]

        # Dense classifier
        out = self.fc1(last_step)
        out = self.relu(out)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class BodySignCNNLSTM(nn.Module):
    """
    Hybrid 1D CNN + LSTM model for sign language detection.
    Extracts temporal/feature correlations with 1D convolution before LSTM.
    """

    def __init__(self, input_size=225, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_type = "cnn_lstm"

        # 1D Convolution over time: expects (batch, channels=features, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        # LSTM taking the conv outputs transposed back to (batch, seq_len, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu_fc = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        # Transpose back to (batch, seq_len, 64) for LSTM
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        out = self.fc1(last_step)
        out = self.relu_fc(out)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def save_model(model, labels, filename="models/video_model.pth", **kwargs):
    """Save model checkpoint with architecture parameters and class labels."""
    checkpoint = {
        "model_state": model.state_dict(),
        "labels": labels,
        "input_size": getattr(model, "input_size", 225),
        "hidden_size": getattr(model, "hidden_size", 128),
        "num_layers": getattr(model, "num_layers", 2),
        "model_type": getattr(model, "model_type", "lstm"),
        "extra": kwargs
    }
    torch.save(checkpoint, filename)


def load_model(filename="models/video_model.pth", device="cpu"):
    """Load model checkpoint and restore architecture and class labels."""
    checkpoint = torch.load(filename, map_location=device)

    labels = checkpoint.get("labels", [])
    input_size = checkpoint.get("input_size", 225)
    hidden_size = checkpoint.get("hidden_size", 128)
    num_layers = checkpoint.get("num_layers", 2)
    model_type = checkpoint.get("model_type", "lstm")
    num_classes = len(labels)

    state = checkpoint.get("model_state", checkpoint)

    # Detect whether it is CNN-LSTM or standard LSTM
    if model_type == "cnn_lstm" or "conv1.weight" in state:
        model = BodySignCNNLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
    else:
        model = BodySignLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, labels


def get_model(input_size=225, hidden_size=128, num_layers=2, num_classes=2, use_cnn=False):
    """Factory function to get model instance."""
    if use_cnn:
        return BodySignCNNLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
    return BodySignLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )
