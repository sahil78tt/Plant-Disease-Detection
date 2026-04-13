# 🌿 Plant Disease Detection using Transfer Learning

An **AI-powered system** that detects diseases in plant leaves using **Transfer Learning with MobileNet**.  
Upload a plant leaf image and get **instant disease predictions**, **confidence scores**, and **remedy suggestions**.

---

# 📌 Project Overview

Plant diseases significantly affect crop yield and food production. This project provides an **automated plant disease detection system** using **Deep Learning and Computer Vision**.

The model uses **Transfer Learning**, where a model trained on **ImageNet** is reused for plant disease classification.

The application allows users to upload a plant leaf image through a **Streamlit web interface** and receive disease predictions.

---

# 🧠 Technologies Used

- Python
- TensorFlow / Keras
- MobileNet (Transfer Learning)
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

# 🌱 Supported Plants & Diseases

| Plant       | Diseases Detected                                             |
| ----------- | ------------------------------------------------------------- |
| Tomato      | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Healthy |
| Potato      | Early Blight, Late Blight, Healthy                            |
| Pepper Bell | Bacterial Spot, Healthy                                       |

Total Classes: **10**

---

# 📁 Project Structure

```
plant-disease-detection/
│
├── dataset/                     # Plant disease image dataset
│
├── models/                      # Saved trained models
│   ├── plant_disease_model.h5
│   ├── class_indices.json
│   └── training_history.png
│
├── train_model.py               # Model training script
├── predict.py                   # Command line prediction script
├── app.py                       # Streamlit web application
├── utils.py                     # Utility functions and constants
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

# ⚙️ Installation

## 1️⃣ Prerequisites

- Python **3.10 / 3.11 / 3.12**
- VS Code (or any editor)
- Minimum **4GB RAM** (8GB recommended)
- At least **2GB free disk space**

Download Python from:

https://www.python.org/downloads/

Verify installation:

```bash
python --version
```

---

# 🖥️ Setup Project

### 1. Open Project in VS Code

```bash
code plant-disease-detection
```

---

### 2. Create Virtual Environment

Open terminal in the project folder.

```bash
python -m venv venv
```

Activate the environment.

**Windows**

```bash
venv\Scripts\activate
```

**Mac / Linux**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

# 📊 Dataset Setup

## Option 1: PlantVillage Dataset (Recommended)

Download dataset from Kaggle:

https://www.kaggle.com/datasets/emmarex/plantdisease

Extract and copy the following folders into the `dataset/` directory:

```
Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_healthy
```

---

## Option 2: Generate Sample Dataset

For quick testing you can generate synthetic images:

```bash
python train_model.py --create-sample
```

---

# 🧪 Model Training

Train the model using the dataset.

```bash
python train_model.py
```

For quick testing:

```bash
python train_model.py --create-sample
```

Example output:

```
============================================================
   PLANT DISEASE DETECTION - MODEL TRAINING
============================================================

Epoch 1/10
loss: 2.30 - accuracy: 0.15 - val_accuracy: 0.17

Epoch 10/10
loss: 0.52 - accuracy: 0.85 - val_accuracy: 0.78

TRAINING COMPLETE
Best Validation Accuracy: 78%
```

---

# 🌐 Run the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

### Features

- Upload plant leaf image
- Detect disease instantly
- View prediction confidence
- Display remedy suggestions
- Confidence chart visualization

---

# 🖥️ Command Line Prediction

Predict disease from an image using terminal.

```bash
python predict.py path/to/image.jpg
```

Or auto-detect a sample image:

```bash
python predict.py
```

---

# 🛠️ Troubleshooting

## Module Not Found Error

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Model Not Found

Run training first:

```bash
python train_model.py --create-sample
```

---

## TensorFlow Installation Error

```bash
pip install tensorflow
```

---

## Streamlit Not Opening

Open manually in browser:

```
http://localhost:8501
```

---

## Out of Memory Error

Reduce batch size in `utils.py`

```python
BATCH_SIZE = 8
```

---

## PowerShell Virtual Environment Issue

Run:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

---

# 🚀 Future Improvements

- Add more plant species (Corn, Apple, Grape)
- Fine-tune MobileNet layers for higher accuracy
- Experiment with EfficientNet / ResNet
- Add Grad-CAM visualization
- Deploy as a mobile app using TensorFlow Lite
- Create REST API using FastAPI
- Add Docker support for deployment

---

# 📚 References

PlantVillage Dataset  
https://www.kaggle.com/datasets/emmarex/plantdisease

MobileNet Paper  
https://arxiv.org/abs/1704.04861

TensorFlow Transfer Learning  
https://www.tensorflow.org/tutorials/images/transfer_learning

Streamlit Documentation  
https://docs.streamlit.io

---

# 👨‍💻 Author

University AI Project  
Built using **Python, TensorFlow, and Streamlit**
