# Radar Target Classification using RCS and Range-Doppler Maps

CNN-based radar target classification system for intelligent target recognition using Range-Doppler (RD) maps, synthetic Radar Cross Section (RCS) simulation data, confidence-based rejection, and occlusion modeling.

---

## Project Overview

This project develops an AI-powered radar target classification framework capable of recognizing:

- Car
- Pedestrian
- Cyclist
- Drone
- Truck
- Tank

The system combines:

- Real radar data from Carrada Dataset
- Synthetic defense target generation using MATLAB RCS simulation
- CNN-based classification
- Occlusion phenomenon modeling
- Confidence-based rejection thresholding
- Temperature scaling calibration
- Interactive frontend for inference

---

## Features

### Radar Target Classification
Classifies radar targets from Range-Doppler maps.

### Multi-Domain Dataset Support
Uses:

- Carrada radar dataset
- Synthetic defense target dataset

### Occlusion Modeling
Artificially introduces partial target masking during training to simulate realistic radar interference.

### Confidence-Based Rejection
Rejects uncertain predictions below a threshold.

### Temperature Scaling Calibration
Improves prediction confidence calibration.

### Interactive Frontend
Upload `.npy` RD maps and obtain:

- Predicted class
- Confidence score
- Acceptance/Rejection decision
- Class probability distribution
- RD visualization

---

## Dataset Structure

```text
data/
├── raw/
│   └── carrada/
│
├── processed/
│
matlab/
└── 04_train_with_sim_data/
    └── sim_data_npy/
        ├── drone/
        ├── truck/
        └── tank/
```

---

## Model Architecture

Input:

```text
256 × 64 × 1 RD Map
```

Pipeline:

```text
Range-Doppler Map
        ↓
CNN Layers
        ↓
Feature Extraction
        ↓
Dense Layers
        ↓
Softmax Classification
        ↓
Confidence Rejection
```

---

## Classes

| Class ID | Target |
|----------|--------|
| 0 | Car |
| 1 | Pedestrian |
| 2 | Cyclist |
| 3 | Drone |
| 4 | Truck |
| 5 | Tank |

---

## Performance

Best recorded run:

| Metric | Value |
|-------|------|
| Test Accuracy | 88.02% |
| Precision | 90.44% |
| Recall | 90.28% |
| F1-score | 90.17% |
| Accepted Accuracy | 95.45% |
| Coverage | 39.69% |

---

## Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/target-classification-rcs.git

cd target-classification-rcs
```

Create environment:

```bash
conda create -n radar python=3.10

conda activate radar
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python -m src.models.train_cnn --config src/config/experiment.yml
```

Outputs:

```text
outputs/
└── cnn_run_xxxxx/
```

---

## Frontend Execution

Run Streamlit:

```bash
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

Upload:

```text
.npy RD maps
```

---

## Project Structure

```text
target-classification-rcs/
│
├── src/
│   ├── models/
│   ├── data/
│   ├── features/
│   ├── utils/
│   └── config/
│
├── matlab/
│
├── frontend/
│
├── outputs/
│
├── requirements.txt
│
└── README.md
```

---

## Software Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Scikit-learn
- Streamlit
- MATLAB
- Matplotlib
- VS Code

---

## Future Improvements

- Transformer-based radar models
- Real-time radar streaming
- Multi-radar fusion
- Advanced attention mechanisms
- Edge deployment optimization

---

## License

MIT License
