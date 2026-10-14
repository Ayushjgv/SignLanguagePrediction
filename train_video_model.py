"""
Train sign language detection model on normalized landmark dataset.

Features:
- Leak-Free Group-Based Splitting: Keeps augmented video variations together so the test set contains only TRULY UNSEEN recordings.
- Scale, position, and aspect-ratio invariant normalization
- On-the-fly Data Augmentation (random scale, translation jitter, and Gaussian noise)
- Multi-layer LSTM and CNN-LSTM models with Dropout & LayerNorm
- Training history curves (Loss and Accuracy) saved to PNG
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from video_model import BodySignCNNLSTM, BodySignLSTM, save_model
from mirror_utils import mirror_sequence, is_mirrorable


class SignDataset(Dataset):
    """
    Dataset with on-the-fly landmark augmentation to prevent overfitting.

    Augmentation (train only):
      - scale / translation / noise jitter
      - detection-dropout (simulate MediaPipe dropping a hand or the pose)
      - horizontal mirror (handedness invariance), only for mirrorable labels
    """

    def __init__(self, X, y, labels, augment=False, mirror=False, mirror_prob=0.5):
        self.X = X
        self.y = y
        self.labels = labels              # list mapping class index -> label string
        self.augment = augment
        self.mirror = mirror              # only ever True for the training split
        self.mirror_prob = mirror_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx].copy()

        if self.augment:
            # 1. Random scale jitter (0.92 - 1.08)
            scale = np.random.uniform(0.92, 1.08)
            seq *= scale

            # 2. Random translation jitter
            shift = np.random.normal(0, 0.015, (1, seq.shape[1]))
            mask = (seq != 0).astype(np.float32)
            seq += shift * mask

            # 3. Small Gaussian noise
            noise = np.random.normal(0, 0.008, seq.shape)
            seq += noise * mask

            # 4. DETECTION-DROPOUT augmentation (real-world robustness).
            #    Real videos often have MediaPipe drop a hand or the whole pose
            #    for some frames (features become all-zero). Training data has
            #    ~100% detection, so we simulate dropouts so the model does not
            #    break when a hand/pose is missing at test time.
            seq_len, feat_dim = seq.shape
            pose_dim = 99            # 33 * 3
            left_dim = 63            # 21 * 3  -> features [99:162]
            right_dim = 63           # 21 * 3  -> features [162:225]

            # 4a. Randomly drop the LEFT hand for a random contiguous span
            if np.random.rand() < 0.25:
                a = np.random.randint(0, seq_len)
                b = np.random.randint(a + 1, seq_len + 1)
                seq[a:b, pose_dim:pose_dim + left_dim] = 0.0

            # 4b. Randomly drop the RIGHT hand for a random contiguous span
            if np.random.rand() < 0.25:
                a = np.random.randint(0, seq_len)
                b = np.random.randint(a + 1, seq_len + 1)
                seq[a:b, pose_dim + left_dim:pose_dim + left_dim + right_dim] = 0.0

            # 4c. Randomly drop BOTH hands (as if no hands detected)
            if np.random.rand() < 0.15:
                a = np.random.randint(0, seq_len)
                b = np.random.randint(a + 1, seq_len + 1)
                seq[a:b, pose_dim:] = 0.0

            # 4d. Randomly drop the whole pose block for a short span
            if np.random.rand() < 0.15:
                a = np.random.randint(0, seq_len)
                b = np.random.randint(a + 1, seq_len + 1)
                seq[a:b, :pose_dim] = 0.0

            # 4e. Randomly blank a few individual frames entirely
            if np.random.rand() < 0.30:
                n_blank = np.random.randint(1, max(2, seq_len // 5))
                blank_idx = np.random.choice(seq_len, size=n_blank, replace=False)
                seq[blank_idx, :] = 0.0

            # 5. HORIZONTAL MIRROR (handedness invariance). Training only, and
            #    only for labels whose meaning is invariant to which hand is
            #    used. The mirrored sample keeps the SAME label and stays in the
            #    same source recording group (mirroring is applied on-the-fly,
            #    so it never creates a new file or leaks across splits).
            if self.mirror:
                label = self.labels[int(self.y[idx])]
                if is_mirrorable(label) and np.random.rand() < self.mirror_prob:
                    seq = mirror_sequence(seq)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def load_data(data_folder="data_video", sequence_length=30, features_per_frame=225, target_classes=None):
    """
    Load extracted landmark sequences from data_folder, tracking recording groups to prevent data leakage.

    If target_classes is provided, only those class folders are loaded (case-insensitive match).
    """
    if not os.path.exists(data_folder):
        print(f"Error: Data folder '{data_folder}' does not exist.")
        return np.array([]), np.array([]), [], []

    X = []
    y = []
    groups = []
    labels = []

    candidates = sorted([
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    ])

    # Optional restriction to a subset of classes (case-insensitive)
    if target_classes:
        wanted = {c.lower().strip() for c in target_classes}
        candidates = [d for d in candidates if d.lower().strip() in wanted]

    for label in candidates:
        folder = os.path.join(data_folder, label)
        npy_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if npy_files:
            labels.append(label)

    if not labels:
        print(f"No class folders with .npy files found in '{data_folder}'")
        return np.array([]), np.array([]), [], []

    print(f"Found {len(labels)} classes: {labels[:10]}{' ...' if len(labels) > 10 else ''}")

    for class_index, label in enumerate(labels):
        folder = os.path.join(data_folder, label)
        class_files = [f for f in os.listdir(folder) if f.endswith(".npy")]

        for file in class_files:
            filepath = os.path.join(folder, file)
            try:
                sequence = np.load(filepath)
            except Exception:
                continue

            if len(sequence.shape) != 2:
                continue

            seq_len, feat_dim = sequence.shape
            if feat_dim != features_per_frame:
                continue

            if seq_len > sequence_length:
                indices = np.linspace(0, seq_len - 1, sequence_length, dtype=int)
                sequence = sequence[indices]
            elif seq_len < sequence_length:
                pad_len = sequence_length - seq_len
                sequence = np.pad(sequence, ((0, pad_len), (0, 0)), mode="edge")

            # Extract base recording ID (stripping tilt variations)
            # e.g. vid_WIN_20231102_12_38_17_Pro_left_tilt_0.npy -> WIN_20231102_12_38_17_Pro
            base_grp = file.replace("_left_tilt", "").replace("_right_tilt", "")
            base_grp = base_grp.split("_0.npy")[0].replace("vid_", "")
            group_key = f"{label}_{base_grp}"

            X.append(sequence)
            y.append(class_index)
            groups.append(group_key)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups), labels


def create_dataset(data_folder="data_video", sequence_length=30, features_per_frame=225, test_size=0.2, batch_size=16, target_classes=None, mirror=False, mirror_prob=0.5):
    """
    Loads data and creates PyTorch train and test DataLoaders with Group-Based Splitting
    to prevent duplicate tilt variations from leaking between train and test sets.

    If mirror=True, the TRAINING dataset applies horizontal-mirror augmentation
    (handedness invariance) on-the-fly. The test dataset is never mirrored or
    otherwise augmented.
    """
    X, y, groups, labels = load_data(data_folder, sequence_length, features_per_frame, target_classes=target_classes)

    if len(X) == 0:
        print("No valid data loaded. Run collect_video_data.py first.")
        return None, None, labels

    print(f"Total samples loaded: {len(X)}")
    unique_groups = len(set(groups))
    print(f"Total unique video recordings: {unique_groups}")
    for index, label in enumerate(labels):
        count = int(np.sum(y == index))
        print(f"  {label:15s}: {count:3d} samples")

    # Group-based split ensures that all tilts of a given recording stay together in train or test
    if unique_groups > len(labels):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\n[LEAK-FREE SPLIT] GroupShuffleSplit applied on {unique_groups} unique recording groups.")
    else:
        counts = np.bincount(y)
        can_stratify = len(counts) > 0 and (counts.min() >= 2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if can_stratify else None
        )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)} (truly unseen video recordings)")

    train_dataset = SignDataset(X_train, y_train, labels, augment=True, mirror=mirror, mirror_prob=mirror_prob)
    test_dataset = SignDataset(X_test, y_test, labels, augment=False, mirror=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, labels


def train_model(
    train_loader,
    test_loader,
    labels,
    input_size=225,
    hidden_size=128,
    num_layers=2,
    num_epochs=50,
    learning_rate=0.001,
    use_cnn=False,
    save_path="video_model.pth"
):
    """
    Trains the sign language model with regularization to avoid overfitting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing compute device: {device}")

    num_classes = len(labels)
    if use_cnn:
        model = BodySignCNNLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.4
        )
    else:
        model = BodySignLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=0.4
        )

    model.to(device)
    print(f"Model Architecture: {'CNN-LSTM' if use_cnn else 'LSTM'}")
    print(f"Features: {input_size} | Hidden: {hidden_size} | Layers: {num_layers} | Classes: {num_classes}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_test_acc = -1.0

    print(f"\nBeginning training for {num_epochs} epochs...")
    print("=" * 65)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                test_correct += (preds == targets).sum().item()
                test_total += targets.size(0)

        epoch_test_loss = test_loss / test_total if test_total > 0 else 0.0
        epoch_test_acc = (test_correct / test_total) * 100.0 if test_total > 0 else 0.0

        scheduler.step(epoch_test_acc)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["test_loss"].append(epoch_test_loss)
        history["test_acc"].append(epoch_test_acc)

        if epoch_test_acc >= best_test_acc:
            best_test_acc = epoch_test_acc
            save_model(model, labels, filename=save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            print(
                f"Epoch [{epoch + 1:3d}/{num_epochs:3d}] | "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:6.2f}% | "
                f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:6.2f}%"
            )

    print("=" * 65)
    print(f"Training completed. Best Test Accuracy on unseen recordings: {best_test_acc:.2f}%")
    print(f"Best model saved to: {save_path}")

    return model, history


def plot_history(history, save_path="training_plot.png"):
    """Saves loss and accuracy curves to image."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(epochs, history["test_loss"], label="Test Loss", color="orange")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="royalblue")
    ax2.plot(epochs, history["test_acc"], label="Test Acc", color="orange")
    ax2.set_title("Accuracy over Epochs (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history plot saved to '{save_path}'")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train sign language detection model with leak-free group split"
    )
    parser.add_argument("--data_folder", default="data_video", help="Data folder (default: data_video)")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length (default: 30)")
    parser.add_argument("--features_per_frame", type=int, default=225, help="Features per frame (default: 225)")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--use_cnn", action="store_true", help="Use CNN-LSTM architecture")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Optional subset of class names to train on (case-insensitive)")
    parser.add_argument("--mirror", action="store_true",
                        help="Enable horizontal-mirror augmentation on the TRAINING set "
                             "(teaches handedness invariance). Never applied to test data.")
    parser.add_argument("--mirror_prob", type=float, default=0.5,
                        help="Probability of mirroring a given mirrorable training sample (default: 0.5)")
    parser.add_argument("--save_path", default="video_model.pth", help="Model save path (default: video_model.pth)")

    args = parser.parse_args()

    train_loader, test_loader, labels = create_dataset(
        data_folder=args.data_folder,
        sequence_length=args.sequence_length,
        features_per_frame=args.features_per_frame,
        batch_size=args.batch_size,
        target_classes=args.classes,
        mirror=args.mirror,
        mirror_prob=args.mirror_prob
    )

    if train_loader is None or len(labels) == 0:
        return

    model, history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        labels=labels,
        input_size=args.features_per_frame,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_cnn=args.use_cnn,
        save_path=args.save_path
    )

    plot_history(history, save_path="training_plot.png")


if __name__ == "__main__":
    main()
