
```markdown
<div align="center">

# 🎥 Human Tracking & Face Recognition System

**Real-time multi-person tracking, face recognition, and human body parsing using state-of-the-art computer vision models.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Demo-orange.svg)]()

*An end-to-end pipeline for intelligent surveillance and evidence collection.*

</div>

---

## 📖 Overview

Traditional surveillance systems simply draw bounding boxes around people. But in real-world scenarios — corporate offices, hospitals, airports — that's not enough. The critical question is:

> **"Who was seen, what were they wearing, where, and when?"**

This project addresses that gap by combining **multi-object tracking**, **face recognition**, and **semantic body parsing** into a single, modular pipeline.

### 🎯 Problems This Project Solves

| Problem | Traditional Approach | This System |
|---------|---------------------|-------------|
| **Identity persistence** | Loses track when people overlap or leave frame | Maintains identity across occlusions using ReID + face embeddings |
| **Evidence collection** | Manual review of hours of footage | Automatic structured storage (face, clothing, timestamp) |
| **Low-quality faces** | Fails at distance or bad angles | Quality gate selects best frames, rejects blurry/noisy input |
| **Search & retrieval** | Rewind and watch | Structured metadata enables fast queries |
| **Body understanding** | Bounding box only | 18-class semantic segmentation (clothing, accessories, body parts) |

---

## ✨ Features

- **🚀 Real-time Multi-Person Tracking** — Powered by YOLOv8 + BoT-SORT with ReID
- **👤 Face Detection & Recognition** — InsightFace (ArcFace) with 512-D embeddings
- **🧬 Human Body Parsing** — FASHN Human Parser with 18 semantic classes
- **🎯 Quality-Aware Frame Selection** — Sharpness, pose, and size scoring to pick the best frames
- **💾 Structured Evidence Storage** — Cropped faces and body parts saved with metadata
- **🔁 Identity Management** — Two-tier threshold logic (confirmed / uncertain / new)
- **📊 Persistent Reports** — JSON logs with full decision history

---

## 🏗️ Architecture

```

┌─────────────────┐
│   Video Input   │
└────────┬────────┘
│
▼
┌─────────────────────────────┐
│  Object Detection (YOLOv8)  │
└────────┬────────────────────┘
│
▼
┌─────────────────────────────┐
│  Tracking (BoT-SORT + ReID) │
└────────┬────────────────────┘
│
┌────┴────┐
▼         ▼
┌────────┐ ┌──────────────────┐
│ Face   │ │ Human Body       │
│ Module │ │ Parser (FASHN)   │
│(Insight│ │ 18 classes       │
│ Face)  │ │                  │
└───┬────┘ └────────┬─────────┘
│               │
▼               ▼
┌─────────────────────────────┐
│   Identity Manager          │
│   (Two-tier matching)       │
└────────┬────────────────────┘
│
▼
┌─────────────────────────────┐
│  Evidence Storage           │
│  (faces, body parts, JSON)  │
└─────────────────────────────┘

```

---

## 🧠 How It Works

### 1. Detection & Tracking
Each frame is processed by **YOLOv8** to detect persons. **BoT-SORT** assigns persistent track IDs, using appearance features (ReID) to maintain identity across occlusions.

### 2. Face Quality Gate
For every detected person, **InsightFace** locates the face and extracts a 512-dimensional embedding. Each face is scored based on:
- **Size** — Minimum bounding box dimensions
- **Sharpness** — Laplacian variance
- **Pose** — Frontal vs. profile angle

Only faces passing the quality threshold are stored in the track's **Feature Bank** (top-K by quality).

### 3. Identity Matching
After collecting enough high-quality samples, the system computes a **weighted average embedding** and matches it against known identities using **cosine similarity**:
- `similarity ≥ 0.65` → **CONFIRMED**
- `0.40 ≤ similarity < 0.65` → **UNCERTAIN** (flagged for review)
- `similarity < 0.40` → **NEW** identity

### 4. Body Parsing & Evidence
**FASHN Human Parser** segments the person into 18 semantic classes (top, pants, skirt, hat, glasses, hands, etc.). Selected parts are saved as transparent PNGs for later analysis.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- ~4GB VRAM for `buffalo_sc` model

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/human-tracking-face-recognition.git
cd human-tracking-face-recognition
pip install -r requirements.txt
```

Requirements

```txt
ultralytics>=8.0.0
insightface>=0.7.3
onnxruntime-gpu>=1.16.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
fashn-human-parser
```

Note: If you don't have a CUDA GPU, replace onnxruntime-gpu with onnxruntime.

Running the Demo

```bash
python src/main.py --video path/to/your/video.mp4 --output output.mp4
```

---

📊 Output Structure

```
output/
├── output.mp4                    # Annotated video
├── mvp_report.json               # Full decision log
├── saved_faces/                  # Cropped faces (high quality)
│   ├── trk0001_face01_q0.85.jpg
│   └── ...
└── saved_body_parts/             # Transparent PNGs of clothing
    ├── trk0001_top_01.png
    ├── trk0001_pants_01.png
    └── ...
```

Report Format

```json
{
  "video": "input.mp4",
  "total_frames": 4200,
  "total_tracks": 87,
  "total_identities": 12,
  "identities": {
    "1": {
      "num_embeddings": 5,
      "tracks": [3, 17, 42],
      "avg_quality": 0.78,
      "face_saved": true,
      "face_path": "saved_faces/trk0003_face01_q0.82.jpg"
    }
  }
}
```

---

⚡ Performance

Tested on NVIDIA RTX 3060 (6GB):

Stage FPS Notes
YOLOv8n + BoT-SORT ~45 Detection only
+ InsightFace (buffalo_sc) ~25 + face recognition
+ FASHN Human Parser ~12 Full pipeline

Performance scales roughly linearly with GPU compute. Higher-end GPUs (RTX 4080, A100) can reach 30-40 FPS on the full pipeline.

---

🛠️ Configuration

All key parameters are in src/config.py:

```python
# Matching thresholds
T_HIGH = 0.65          # Confirmed match
T_LOW = 0.40           # Uncertain match

# Face quality gates
FACE_MIN_SIZE = 30     # Min face size (px)
FACE_MIN_SHARPNESS = 25.0

# Storage
FACE_PADDING = 0.30
FACE_UPSCALE = 4
```

---

⚠️ Limitations

This project is a proof of concept. Known limitations:

· Identity fragmentation in dense crowds — BoT-SORT may assign new IDs after long occlusions
· Distance sensitivity — Face recognition degrades beyond ~5-7 meters with typical cameras
· Lighting dependence — Quality gate may reject valid faces in poor lighting
· Cross-camera tracking — Not implemented in this demo (single-camera only)
· No gallery pre-registration — System learns incrementally from observation

---

🗺️ Roadmap

☐ Cross-camera identity association with geometric matching
☐ Integration with vector database (Milvus/FAISS) for large-scale storage
☐ Semantic search via CLIP-based text queries
☐ Web dashboard for live monitoring
☐ Support for RTSP streams

---

🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

---