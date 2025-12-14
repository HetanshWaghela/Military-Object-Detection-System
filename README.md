# 🎯 Military Object Detection with YOLOv8

Real-time multi-class object detection system for military imagery using YOLOv8s.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | 42.6% |
| Precision | 60.3% |
| Recall | 41.0% |
| Inference | ~5ms (GPU) |

### Top Performing Classes
| Class | AP@0.5 |
|-------|--------|
| Military Aircraft | **82.1%** |
| Military Tank | **81.6%** |
| Camouflage Soldier | 67.4% |
| Soldier | 62.9% |

## 📁 Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Data preparation
│   ├── 03_training.ipynb         # Model training (Colab-ready)
│   ├── 04_evaluation.ipynb       # Performance evaluation
│   ├── 05_inference.ipynb        # Run predictions
│   └── 06_advanced_analysis.ipynb # Error analysis
├── config/
│   ├── dataset.yaml              # Dataset configuration
│   └── training_configs.json     # Training hyperparameters
├── models/
│   └── best_model.pt             # Trained weights
├── results/
│   └── evaluation_results.json   # Metrics
└── figures/                      # Visualizations
```

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/military-object-detection.git
cd military-object-detection

# Install dependencies
pip install ultralytics torch torchvision opencv-python matplotlib pandas tqdm
```

## 🏃 Quick Start

### Training (Colab Recommended)
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data='config/dataset.yaml',
    epochs=20,
    batch=16,
    imgsz=640,
    patience=5
)
```

### Inference
```python
from ultralytics import YOLO

model = YOLO('models/best_model.pt')
results = model.predict('path/to/image.jpg', conf=0.25)
```

## 📊 Dataset

12-class military object detection dataset:

| Category | Classes |
|----------|---------|
| Personnel | camouflage_soldier, soldier, civilian |
| Ground Vehicles | military_tank, military_truck, military_vehicle, civilian_vehicle |
| Equipment | weapon, military_artillery, trench |
| Aerial/Naval | military_aircraft, military_warship |

**Splits:** Train (8,253) / Val (1,766) / Test (1,396)

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s (11.2M params) |
| Epochs | 20 (early stopped at 11) |
| Batch Size | 16 |
| Image Size | 640×640 |
| Optimizer | AdamW |
| Learning Rate | 0.002 |

## 🔍 Key Findings

1. **Strong on distinct objects**: Aircraft and tanks achieve >80% AP due to unique visual features
2. **Class imbalance impact**: Rare classes (<50 samples) show poor performance
3. **Efficient inference**: Real-time capable on GPU, operational on CPU

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda` | Dataset exploration, class distribution, visualizations |
| `02_preprocessing` | Data validation, augmentation setup |
| `03_training` | Model training with YOLOv8 |
| `04_evaluation` | mAP calculation, per-class metrics |
| `05_inference` | Generate predictions on test set |
| `06_advanced_analysis` | Error analysis, confusion patterns |

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
# Military-Object-Detection-System
