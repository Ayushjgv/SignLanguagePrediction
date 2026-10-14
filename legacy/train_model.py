import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model import HandSignLSTM, save_model


def load_data():
    X = []
    y = []
    labels = []

    for label in sorted(os.listdir("data")):
        if os.path.isdir(os.path.join("data", label)):
            labels.append(label)

    for class_index, label in enumerate(labels):
        folder = os.path.join("data", label)

        for file in os.listdir(folder):
            if file.endswith(".npy"):
                sequence = np.load(os.path.join(folder, file))

                if sequence.shape == (30, 63):
                    X.append(sequence)
                    y.append(class_index)

    return np.array(X, dtype=np.float32), np.array(y), labels


def main():
    X, y, labels = load_data()

    if len(X) == 0:
        print("No data found. Run collect_data.py first.")
        return

    if len(labels) < 2:
        print("You need at least 2 classes.")
        return

    print("Classes:", labels)
    print("Total samples:", len(X))

    for index, label in enumerate(labels):
        print(label, ":", np.sum(y == index), "samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_data = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    model = HandSignLSTM(input_size=63, hidden_size=64, num_classes=len(labels))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += len(targets)

        train_accuracy = correct / total * 100
        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {average_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += len(targets)

    test_accuracy = correct / total * 100
    print("\nTest Accuracy:", round(test_accuracy, 2), "%")

    save_model(model, labels, "model.pth")
    print("Model saved as model.pth")


if __name__ == "__main__":
    main()
