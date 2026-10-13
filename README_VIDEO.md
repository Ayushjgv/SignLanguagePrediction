# Video Dataset Sign Language Detection Pipeline

This pipeline trains and tests a deep learning model for sign language detection using **full body pose + hand landmarks** from video files in `Video_Dataset`.

---

## 📌 Architecture & Features

- **MediaPipe Pose**: 33 full-body landmarks $\times$ 3 coordinates $(x, y, z) = 99$ features
- **MediaPipe Hands**: 21 landmarks $\times$ 2 hands $\times$ 3 coordinates $(x, y, z) = 126$ features
- **Total Input**: **225 features per frame**, sampled over a sequence of **30 frames** $(30 \times 225)$
- **Models**: Multi-layer LSTM / CNN-LSTM with LayerNorm and Dropout for stability
- **Inference**:
  - **Video File Testing**: Test on pre-recorded videos with live playback or video export
  - **Real-time Webcam Testing**: Continuous live detection with 30-frame rolling window and smoothing

---

## 🗂 Project Workflow & Scripts

| Script | Purpose |
| :--- | :--- |
| `collect_video_data.py` | Extracts pose & hand landmarks from `Video_Dataset/` and saves `.npy` sequences |
| `train_video_model.py` | Trains the sequence classification model and saves `video_model.pth` |
| `test_video.py` | Tests the trained model on video files with visual playback or export |
| `realtime_detect.py` | Runs real-time live sign detection from your webcam |
| `video_model.py` | PyTorch model definitions (`BodySignLSTM`, `BodySignCNNLSTM`) |

---

## 🚀 How to Run (Step-by-Step)

### Step 1: Extract Landmark Data from `Video_Dataset`

Run extraction on all classes (now **parallelized** across CPU cores for speed):
```powershell
.\.venv\Scripts\python.exe collect_video_data.py --video_folder Video_Dataset --output_folder data_video --workers 8
```

**Optional:** Extract only a subset of signs (fast), e.g. the first 20 classes:
```powershell
.\.venv\Scripts\python.exe collect_video_data.py --video_folder Video_Dataset --output_folder data_video --classes Bear Break Brinjal Budget Busy Cabbage Carrot Cauliflower Chilli Clean Close Come Cook Crocodile Cry Cucumber Deer Drink Elephant Exam --workers 8
```

Extraction auto-resumes (already-extracted videos are skipped) and runs each worker
with its own MediaPipe detectors, so it scales with the number of CPU cores.

---

### Step 2: Train the Model

Train on **all extracted classes**:
```powershell
.\.venv\Scripts\python.exe train_video_model.py --data_folder data_video --epochs 50 --batch_size 16 --save_path video_model.pth
```

**Train on a specific subset of classes** (e.g. the first 20):
```powershell
.\.venv\Scripts\python.exe train_video_model.py --data_folder data_video --classes Bear Break Brinjal Budget Busy Cabbage Carrot Cauliflower Chilli Clean Close Come Cook Crocodile Cry Cucumber Deer Drink Elephant Exam --epochs 60 --batch_size 16 --save_path video_model.pth
```

**Training Options:**
- `--use_cnn`: Train a hybrid CNN-LSTM model
- `--classes A B C`: Restrict training to specific class folders (case-insensitive)
- `--epochs 60`: Number of training epochs (default: 50)
- `--batch_size 16`: Batch size (default: 16)
- `--hidden_size 128`: Number of LSTM units (default: 128)

> **Important:** The model can only predict classes it was trained on. If you test a
> video of a sign that was not included in training, it will be forced to guess among
> the trained classes. Make sure the classes you care about are extracted (Step 1)
> **and** included in training (Step 2).

The best model checkpoint and a training plot (`training_plot.png`) will be automatically saved.

---

### Step 2.5: Evaluate the Model (optional)

Report held-out accuracy (on truly unseen recordings) and a per-class breakdown:
```powershell
.\.venv\Scripts\python.exe evaluate_model.py --model video_model.pth
```

Quickly predict the label of individual videos:
```powershell
.\.venv\Scripts\python.exe test_predictions.py
```


---

### Step 3: Test Model on Video Files

To test on a recorded video and view the live playback with sign prediction overlay:
```powershell
.\.venv\Scripts\python.exe test_video.py Sample_Videos/Bear.mp4 --model video_model.pth --show
```

To test and save the annotated video to a file:
```powershell
.\.venv\Scripts\python.exe test_video.py Sample_Videos/Hello.mp4 --model video_model.pth --output_video result.mp4
```

---

### Step 4: Test Model in Real-Time (Webcam)

Launch real-time webcam detection:
```powershell
.\.venv\Scripts\python.exe realtime_detect.py --model video_model.pth --camera 0
```

**Interactive Controls During Webcam Testing:**
- `q` or `ESC`: Exit
- `c`: Clear buffer / reset detection
- `h`: Toggle landmark skeleton drawing on/off
