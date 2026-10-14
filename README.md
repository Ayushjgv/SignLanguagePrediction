# SignLanguageMe — Sign Language Recognition (Pose + Both Hands)

Real-time / video sign language recognition using **MediaPipe** full-body pose
(33 landmarks) + **both hands** (21 landmarks each), fed to a **PyTorch LSTM**
sequence classifier. 225 normalized features per frame, 30-frame sequences.

The feature pipeline is **scale-, position-, and aspect-ratio invariant**, and
training includes **detection-dropout** and **horizontal-mirror** augmentation so
the model is robust to missing detections and to which hand is used.

---

## Project structure

```
SignLanguageMe-v1/
├── README.md                 # this file
├── requirements.txt
├── .gitignore
├── .venv/                    # Python virtual environment (not committed)
│
├── src/                      # all source code
│   ├── landmark_utils.py     #   feature extraction + normalization + skeleton drawing
│   ├── mirror_utils.py       #   safe horizontal-mirror augmentation + label allowlist
│   ├── video_model.py        #   LSTM / CNN-LSTM model, save/load
│   ├── collect_video_data.py #   Step 1: extract .npy features from videos (parallel)
│   ├── train_video_model.py  #   Step 2: train the model (leak-free group split)
│   ├── test_video.py         #   Step 3: test on a video file (playback / export)
│   ├── realtime_detect.py    #   Step 4: real-time webcam detection
│   ├── evaluate_model.py     #   held-out accuracy + per-class report
│   ├── check_classes.py      #   inspect classes across model / data / source
│   ├── test_mirror.py        #   unit tests for mirror augmentation (run these)
│   ├── test_handedness.py    #   verify left/right-hand invariance of a model
│   └── test_predictions.py   #   quick per-class prediction sanity check
│
├── models/                   # trained checkpoints
│   ├── video_model.pth           # trained WITHOUT mirror augmentation
│   └── video_model_mirror.pth    # trained WITH mirror augmentation (recommended)
│
├── data_video/               # extracted features: one folder per class (.npy)
├── Video_Dataset/            # raw source videos: one folder per class
├── test_videos/              # your own test videos
├── output/                   # generated artifacts (training_plot.png, poster, result videos)
└── legacy/                   # archived old hand-only 3-class pipeline (fine/hello/idle)
    ├── collect_data.py  train_model.py  model.py  model.pth  detect.py
    ├── data/            # old 3-class features
    └── models/          # v1.pth, v2.pth
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> Always run commands **from the project root** (the folder containing `src/`),
> because the scripts use paths relative to the current directory, e.g.
> `python src\test_video.py ...`.

---

## Workflow

### 1. Extract features from videos
```powershell
.\.venv\Scripts\python.exe src\collect_video_data.py --video_folder Video_Dataset --output_folder data_video --workers 8
```
- Parallel across CPU cores; auto-resumes (skips already-extracted videos).
- `--classes A B C` to extract only some signs; `--augment` for extra sequences.

### 2. Train
```powershell
.\.venv\Scripts\python.exe src\train_video_model.py --data_folder data_video --epochs 70 --batch_size 16 --mirror --mirror_prob 0.5 --save_path models/video_model_mirror.pth
```
- `--mirror`: **recommended** — teaches the model a sign is the same regardless of
  which hand is used (handedness invariance). Applied to training data only.
- `--classes A B C`: train on a subset.
- Saves the best checkpoint (by held-out accuracy) and `output/training_plot.png`.

### 3. Test on a video
```powershell
.\.venv\Scripts\python.exe src\test_video.py test_videos/drink.mp4 --model models/video_model_mirror.pth --show
.\.venv\Scripts\python.exe src\test_video.py test_videos/drink.mp4 --model models/video_model_mirror.pth --output_video output/result.mp4
```

### 4. Real-time webcam
```powershell
.\.venv\Scripts\python.exe src\realtime_detect.py --model models/video_model_mirror.pth --camera 0
```

> **Realtime accuracy tip — the buffer matters.** The model was trained on 30
> frames spread across the *whole* gesture (4–7s). A 30-*consecutive*-frame
> buffer only covers ~1s and gives poor accuracy. `realtime_detect.py` therefore
> keeps a longer history (`--buffer`, default **75** frames ≈ 2.5s) and samples
> 30 frames across it for each prediction — this matches training and is the
> single biggest realtime improvement. Tune with `--buffer 60..90` and
> `--stride 3` (infer every N frames). Hold a sign steady for ~2–3 seconds.

To test the realtime logic on a video file (no webcam needed):
```powershell
.\.venv\Scripts\python.exe src\simulate_realtime.py --video test_videos/drink.mp4 --expect drink
```

### Evaluate / inspect
```powershell
.\.venv\Scripts\python.exe src\evaluate_model.py --model models/video_model_mirror.pth
.\.venv\Scripts\python.exe src\check_classes.py --model models/video_model_mirror.pth
```

### Tests
```powershell
.\.venv\Scripts\python.exe src\test_mirror.py            # mirror unit tests
.\.venv\Scripts\python.exe src\test_handedness.py --model models/video_model_mirror.pth
```

---

## Which model should I use?

| Model | Trained with | Use when |
|-------|--------------|----------|
| `models/video_model_mirror.pth` | mirror augmentation | **default** — works with either hand |
| `models/video_model.pth` | no mirror | baseline / comparison |

Both are trained on all 61 classes and reach ~99% accuracy on held-out
recordings **from the training source**.

---

## Honest limitations

- The dataset (`Video_Dataset/`) is a **single signer**. Your own `test_videos/`
  (different person, portrait phone) are a different domain, so a few look-alike
  pairs (e.g. *drink* vs *uncle*) can still be confused on borderline clips.
- Mirror augmentation fixes **handedness** but is **not** a substitute for
  recording new signers. To improve real-world accuracy, add a few of your own
  clips per sign to `Video_Dataset/` and retrain.
