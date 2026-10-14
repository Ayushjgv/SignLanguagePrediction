"""
Generate a comprehensive set of statistics figures into ./images/ .

Covered figures
---------------
  data      : class distribution, leak-free split, feature/hand detection
  model     : architecture diagram, parameter breakdown
  training  : loss & accuracy curves (from output/training_history.json)
  evaluation: confusion matrix, per-class accuracy, per-class P/R/F1,
              metrics summary, top confusions
  robustness: plain-vs-mirror model comparison, learned-feature t-SNE,
              handedness invariance (real videos), realtime buffer strategy

Run from the project root:
    .\\.venv\\Scripts\\python.exe src\\generate_stats.py
"""
import os
import glob
import json

import numpy as np
import torch
import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score, balanced_accuracy_score)
from sklearn.manifold import TSNE

from video_model import load_model
from landmark_utils import extract_normalized_features
from mirror_utils import mirror_sequence
from train_video_model import load_data

IMG = "images"
HISTORY_JSON = os.path.join("output", "training_history.json")
TEST_VIDEOS = sorted(glob.glob(os.path.join("test_videos", "*.mp4")))
MIRROR_MODEL = "models/video_model_mirror.pth"
PLAIN_MODEL = "models/video_model.pth"
SEQ_LEN = 30

os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})

# consistent, pleasant palette
COLORS = ["#2b7bba", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
          "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 1-3. Data statistics
# ---------------------------------------------------------------------------
def plot_class_distribution(y, labels):
    counts = np.bincount(y, minlength=len(labels))
    order = np.argsort(counts)[::-1]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(labels)), counts[order],
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"Class distribution  —  {len(labels)} classes, "
                 f"{len(y)} sequences total")
    # reference line for the mean
    ax.axhline(counts.mean(), color="crimson", ls="--", lw=1,
               label=f"mean = {counts.mean():.1f}")
    ax.legend(loc="upper right")
    save(fig, "01_class_distribution.png")


def plot_split_stats(y, groups, labels):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    tr_g = len(set(groups[tr]))
    te_g = len(set(groups[te]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["Train", "Test"], [len(tr), len(te)],
                color=[COLORS[0], COLORS[1]])
    axes[0].set_title("Sequences per split")
    for i, v in enumerate([len(tr), len(te)]):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    axes[1].bar(["Train", "Test"], [tr_g, te_g], color=[COLORS[0], COLORS[1]])
    axes[1].set_title(f"Unique recordings (groups) per split  —  "
                      f"{len(set(groups))} total")
    for i, v in enumerate([tr_g, te_g]):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

    fig.suptitle("Leak-free GroupShuffleSplit (test_size=0.2, random_state=42) "
                 "— all tilts of a recording stay in one split", y=1.02, fontsize=10)
    save(fig, "02_group_split.png")


def plot_feature_stats(X, y, labels):
    # fraction of non-zero landmark values => detection / activation proxy
    hand_nz = (X[:, :, 99:] != 0).mean(axis=(1, 2))
    pose_nz = (X[:, :, :99] != 0).mean(axis=(1, 2))

    hand_c = [hand_nz[y == i].mean() if np.any(y == i) else 0
              for i in range(len(labels))]
    pose_c = [pose_nz[y == i].mean() if np.any(y == i) else 0
              for i in range(len(labels))]

    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(idx - 0.2, pose_c, width=0.4, label="Pose activation", color=COLORS[0])
    ax.bar(idx + 0.2, hand_c, width=0.4, label="Hands activation", color=COLORS[3])
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Fraction of non-zero features")
    ax.set_ylim(0, 1.05)
    ax.set_title("Feature detection / activation per class "
                 "(non-zero landmark fraction)")
    ax.legend()
    save(fig, "03_feature_detection.png")



# ---------------------------------------------------------------------------
# 4-6. Model architecture, parameters, training curves
# ---------------------------------------------------------------------------
def plot_architecture(model):
    hidden = model.hidden_size
    num_layers = model.num_layers
    ncls = model.num_classes
    head = hidden // 2

    stages = [
        ("Input   (30 timesteps x 225 features)", "(B, 30, 225)", "#dfeaf6"),
        (f"{num_layers}-layer LSTM   (hidden = {hidden})",
         f"(B, 30, {hidden})  ->  last step (B, {hidden})", "#cfe2f3"),
        (f"Linear {hidden}->{head}  + ReLU + LayerNorm + Dropout(0.3)",
         f"(B, {head})", "#e8f4d9"),
        (f"Linear {head}->{ncls}", f"(B, {ncls})", "#fbe3c8"),
        (f"Softmax  ->  {ncls} classes", f"(B, {ncls})  probability", "#f6d4d4"),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.axis("off")

    n = len(stages)
    box_h, gap = 0.85, 0.5
    xs, ws = 1.0, 6.0
    top = n * box_h + (n - 1) * gap
    y = top

    for i, (title, shape, fc) in enumerate(stages):
        yy = y - box_h
        ax.add_patch(FancyBboxPatch((xs, yy), ws, box_h, boxstyle="round,pad=0.05",
                                    fc=fc, ec="black", lw=1.3))
        ax.text(xs + ws / 2, yy + box_h / 2, title, ha="center", va="center", fontsize=10)
        ax.text(xs + ws + 0.4, yy + box_h / 2, shape, ha="left", va="center",
                fontsize=9, color="#444", style="italic")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((xs + ws / 2, yy),
                                         (xs + ws / 2, yy - gap),
                                         arrowstyle="-|>", mutation_scale=20,
                                         lw=1.8, color="#333"))
        y = yy - gap

    ax.set_ylim(-gap, top + 0.2)
    ax.set_title("Model architecture  —  BodySignLSTM", fontsize=13, loc="left")
    save(fig, "04_model_architecture.png")


def plot_param_table(model):
    total = sum(p.numel() for p in model.parameters())
    groups = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()

    keys = list(groups.keys())
    vals = [groups[k] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(keys, vals, color=[COLORS[i % len(COLORS)] for i in range(len(keys))])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Trainable parameters")
    ax.set_title(f"Parameters by module   (total = {total:,})")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    save(fig, "05_model_parameters.png")


def plot_training_curves():
    if not os.path.exists(HISTORY_JSON):
        print(f"  [skip] {HISTORY_JSON} not found (training not finished?)")
        return
    with open(HISTORY_JSON) as f:
        h = json.load(f)

    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    axes[0].plot(ep, h["train_loss"], label="Train loss", color=COLORS[0])
    axes[0].plot(ep, h["test_loss"], label="Test loss", color=COLORS[1])
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(ep, h["train_acc"], label="Train accuracy", color=COLORS[0])
    axes[1].plot(ep, h["test_acc"], label="Test accuracy", color=COLORS[1])
    best = float(max(h["test_acc"]))
    axes[1].axhline(best, color="crimson", ls="--", lw=1)
    axes[1].text(len(ep), best, f" best = {best:.2f}%", va="center", ha="right",
                 fontsize=8, color="crimson")
    axes[1].set_title("Accuracy over epochs (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    fig.suptitle("Training dynamics  —  61-class LSTM with mirror augmentation",
                 y=1.03, fontsize=11)
    save(fig, "06_training_curves.png")


# ---------------------------------------------------------------------------
# 7-10, 14. Held-out evaluation
# ---------------------------------------------------------------------------
def eval_model(model, X, y, groups, ncls):
    model.eval()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te = next(gss.split(np.zeros(len(y)), y, groups))
    Xte, yte = X[te], y[te]
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    yhat = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    lidx = np.arange(ncls)
    p, r, f, _ = precision_recall_fscore_support(yte, yhat, average=None,
                                                 labels=lidx, zero_division=0)
    _, _, f_macro, _ = precision_recall_fscore_support(yte, yhat, average="macro", zero_division=0)
    _, _, f_weight, _ = precision_recall_fscore_support(yte, yhat, average="weighted", zero_division=0)
    return {
        "idx": te, "y": yte, "yhat": yhat, "probs": probs,
        "acc": accuracy_score(yte, yhat),
        "bal_acc": balanced_accuracy_score(yte, yhat),
        "top5": float((top5 == yte[:, None]).any(axis=1).mean()),
        "f_macro": f_macro, "f_weight": f_weight,
        "per_p": p, "per_r": r, "per_f": f,
    }


def plot_confusion(yte, yhat, labels):
    cm = confusion_matrix(yte, yhat, labels=np.arange(len(labels))).astype(float)
    norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized) on unseen recordings")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="row fraction")
    save(fig, "07_confusion_matrix.png")


def plot_per_class_acc(yte, yhat, labels):
    cm = confusion_matrix(yte, yhat, labels=np.arange(len(labels))).astype(float)
    support = cm.sum(axis=1)
    acc = np.where(support > 0, cm.diagonal() / np.maximum(support, 1), 0)
    order = np.argsort(acc)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(labels)), acc[order], color=COLORS[0])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class accuracy")
    mean_acc = acc[support > 0].mean()
    ax.axhline(mean_acc, color="crimson", ls="--", lw=1, label=f"mean = {mean_acc:.3f}")
    ax.set_title("Per-class accuracy (sorted) on unseen recordings")
    ax.legend()
    save(fig, "08_per_class_accuracy.png")


def plot_per_class_prf(p, r, f, labels):
    idx = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(idx, p, label="Precision", color=COLORS[0], lw=1.3)
    ax.plot(idx, r, label="Recall", color=COLORS[1], lw=1.3)
    ax.plot(idx, f, label="F1", color=COLORS[2], lw=1.3)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    save(fig, "09_per_class_prf.png")


def plot_metrics_summary(m):
    names = ["Overall\naccuracy", "Balanced\naccuracy", "Macro F1",
             "Weighted F1", "Top-5\naccuracy"]
    vals = [m["acc"], m["bal_acc"], m["f_macro"], m["f_weight"], m["top5"]]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(names, vals, color=[COLORS[i] for i in range(len(names))])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v*100:.2f}%",
                ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation metrics on truly unseen recordings")
    save(fig, "10_metrics_summary.png")


def plot_top_confusions(yte, yhat, labels, k=20):
    cm = confusion_matrix(yte, yhat, labels=np.arange(len(labels))).astype(float)
    np.fill_diagonal(cm, 0)
    pairs = np.argsort(cm.ravel())[::-1][:k]
    rows = []
    for idx in pairs:
        t, pr = divmod(int(idx), len(labels))
        if cm[t, pr] > 0:
            rows.append((labels[t], labels[pr], int(cm[t, pr])))
    if not rows:
        print("  [skip] no confusions found")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[2] for r in rows], color=COLORS[3])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{a}  ->  {b}" for a, b, _ in rows], fontsize=8)
    ax.set_xlabel("Times misclassified")
    ax.set_title(f"Top {len(rows)} confusions (true -> predicted)")
    save(fig, "14_top_confusions.png")


# ---------------------------------------------------------------------------
# 11. Model comparison ; 15. t-SNE of learned features
# ---------------------------------------------------------------------------
def plot_model_comparison(X, y, groups, ncls):
    results = {}
    for name, path in [("Plain model", PLAIN_MODEL), ("Mirror model", MIRROR_MODEL)]:
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        mdl, _ = load_model(path)
        results[name] = eval_model(mdl, X, y, groups, ncls)

    if not results:
        return None
    metrics = ["acc", "bal_acc", "f_macro", "top5"]
    mlabels = ["Accuracy", "Balanced acc", "Macro F1", "Top-5 acc"]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (name, m) in enumerate(results.items()):
        off = (i - (len(results) - 1) / 2) * w
        vals = [m[k] for k in metrics]
        ax.bar(x + off, vals, w, label=name, color=COLORS[i])
        for j, v in enumerate(vals):
            ax.text(x[j] + off, v, f"{v*100:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on identical unseen recordings")
    ax.legend()
    save(fig, "11_model_comparison.png")
    return results


def plot_tsne(model, Xte, yte):
    captured = {}

    def hook(mod, inp, out):
        captured["f"] = out.detach().numpy()

    h = model.norm.register_forward_hook(hook)
    with torch.no_grad():
        model(torch.tensor(Xte, dtype=torch.float32))
    h.remove()
    feat = captured["f"]

    emb = TSNE(n_components=2, random_state=42, perplexity=30,
               init="pca").fit_transform(feat)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=yte, cmap="tab20", s=14, alpha=0.85)
    ax.set_title("t-SNE of learned 64-D features (mirror model, unseen recordings)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.colorbar(scatter, ax=ax, label="class id")
    save(fig, "15_tsne_features.png")


# ---------------------------------------------------------------------------
# 12-13. Realtime robustness using real videos (MediaPipe)
# ---------------------------------------------------------------------------
def make_detectors():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose, hands


def extract_video(path, pose, hands):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # same as realtime (selfie view)
        feats, _, _ = extract_normalized_features(frame, pose, hands)
        frames.append(feats)
    cap.release()
    return np.asarray(frames, dtype=np.float32)  # (L, 225)


def sample30(seq):
    L = len(seq)
    if L >= SEQ_LEN:
        idx = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
        out = seq[idx]
    else:
        out = np.pad(seq, ((0, SEQ_LEN - L), (0, 0)), mode="edge")
    return out.astype(np.float32)


def predict(model, seq30):
    with torch.no_grad():
        logits = model(torch.tensor(seq30[None], dtype=torch.float32))
        return int(torch.argmax(logits, dim=1).item())


def plot_handedness(models, vid):
    names = sorted(vid.keys())
    match = {mname: [] for mname in models}
    for stem in names:
        s30 = sample30(vid[stem])
        m30 = mirror_sequence(s30)
        for mname, mdl in models.items():
            same = predict(mdl, s30) == predict(mdl, m30)
            match[mname].append(1 if same else 0)

    x = np.arange(len(names))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, mname in enumerate(models):
        off = (i - (len(models) - 1) / 2) * w
        vals = match[mname]
        rate = float(np.mean(vals)) if vals else 0.0
        ax.bar(x + off, vals, w, label=f"{mname} (mean {rate:.0%})",
               color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks([0, 1])
    ax.set_ylabel("Mirror-invariant prediction")
    ax.set_title("Handedness invariance: pred(original) == pred(mirrored) per video")
    ax.legend(loc="lower right")
    save(fig, "12_handedness_invariance.png")


def plot_buffer(model, valid, labels):
    buffers = [30, 45, 60, 75, 90]
    accs = []
    for B in buffers:
        correct = 0
        for seq, exp in valid:
            sub = seq[-B:] if len(seq) >= B else seq  # last B frames (streaming)
            if predict(model, sample30(sub)) == exp:
                correct += 1
        accs.append(correct / len(valid))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(buffers, accs, marker="o", color=COLORS[0], lw=2.2, zorder=2)
    for b, a in zip(buffers, accs):
        ax.annotate(f"{a*100:.0f}%", (b, a), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold")
    ax.axvline(30, color="crimson", ls="--", lw=1)
    ax.set_xlabel("History buffer size (frames)")
    ax.set_ylabel("Accuracy on test videos")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(buffers)
    ax.set_title("Realtime buffer size vs accuracy  "
                 "(30 frames sampled across the buffer)")
    save(fig, "13_buffer_strategy.png")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def plot_video_stats(labels, mirror_model):
    if not TEST_VIDEOS:
        print("  [skip] no test_videos/*.mp4 found")
        return

    pose, hands = make_detectors()
    vid = {}
    for path in TEST_VIDEOS:
        stem = os.path.splitext(os.path.basename(path))[0]
        seq = extract_video(path, pose, hands)
        if len(seq) >= 10:
            vid[stem] = seq
        print(f"    extracted {stem}: {len(seq)} frames")
    pose.close()
    hands.close()

    if not vid:
        print("  [skip] no usable test videos")
        return

    models = {"Mirror": mirror_model}
    if os.path.exists(PLAIN_MODEL):
        pm, _ = load_model(PLAIN_MODEL)
        pm.eval()
        models = {"Plain": pm, "Mirror": mirror_model}
    plot_handedness(models, vid)

    lower = {l.lower(): i for i, l in enumerate(labels)}
    valid = [(seq, lower[stem.lower()]) for stem, seq in vid.items()
             if stem.lower() in lower]
    if valid:
        print(f"    buffer test on {len(valid)} videos matching a class label")
        plot_buffer(mirror_model, valid, labels)
    else:
        print("  [skip] no video filename matched a class label for buffer test")


def main():
    print("Loading dataset ...")
    X, y, groups, labels = load_data(data_folder="data_video")
    if len(X) == 0:
        print("No data found in data_video/.")
        return
    ncls = len(labels)
    print(f"  {len(X)} sequences, {ncls} classes, {len(set(groups))} groups")

    print("\n[1/5] Data statistics")
    plot_class_distribution(y, labels)
    plot_split_stats(y, groups, labels)
    plot_feature_stats(X, y, labels)

    print("\n[2/5] Model architecture + parameters")
    mirror_model, _ = load_model(MIRROR_MODEL)
    mirror_model.eval()
    plot_architecture(mirror_model)
    plot_param_table(mirror_model)

    print("\n[3/5] Held-out evaluation (mirror model)")
    m = eval_model(mirror_model, X, y, groups, ncls)
    Xte = X[m["idx"]]
    print(f"    accuracy={m['acc']*100:.2f}%  bal_acc={m['bal_acc']*100:.2f}%  "
          f"macroF1={m['f_macro']:.3f}  top5={m['top5']*100:.2f}%")
    plot_confusion(m["y"], m["yhat"], labels)
    plot_per_class_acc(m["y"], m["yhat"], labels)
    plot_per_class_prf(m["per_p"], m["per_r"], m["per_f"], labels)
    plot_metrics_summary(m)
    plot_top_confusions(m["y"], m["yhat"], labels)
    plot_tsne(mirror_model, Xte, m["y"])

    print("\n[4/5] Model comparison + training curves")
    plot_model_comparison(X, y, groups, ncls)
    plot_training_curves()

    print("\n[5/5] Realtime robustness (real videos via MediaPipe)")
    plot_video_stats(labels, mirror_model)

    print(f"\nDone. All figures written to ./{IMG}/")


if __name__ == "__main__":
    main()




