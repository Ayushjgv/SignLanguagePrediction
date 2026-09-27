import torch
import torch.nn as nn


class HandSignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)


def save_model(model, labels, filename="model.pth"):
    torch.save({
        "model_state": model.state_dict(),
        "labels": labels
    }, filename)


def load_model(filename="model.pth"):
    checkpoint = torch.load(filename, map_location="cpu")
    labels = checkpoint["labels"]
    model = HandSignLSTM(num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, labels
