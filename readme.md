🐬 Lightweight Deep Learning for Marine Object Classification

A web-based application that classifies marine organisms using lightweight deep learning models and provides real-time predictions with confidence scores. The system compares multiple CNN architectures to analyze performance, efficiency, and transparency.

📌 Project Overview

Marine object classification is essential for marine research and conservation. Traditional manual methods are slow and not scalable, while heavy deep learning models require high computational resources.

This project introduces a lightweight deep learning-based web application that:

Classifies marine images into five categories

Uses efficient CNN models for faster inference

Displays confidence scores for transparency

Compares multiple model performances

🎯 Marine Categories

Dolphin

Fish

Lobster

Octopus

Sea Horse

🧠 Models Implemented
Model	Type	Training Method
MobileNetV2	Lightweight CNN	Transfer Learning
EfficientNet-B0	Lightweight CNN	Transfer Learning
Custom CNN	Simple CNN	Trained from scratch
⚙️ Features

✔ Image upload via web interface
✔ Real-time classification
✔ Confidence score display
✔ Model-wise comparison
✔ Lightweight & fast inference
✔ Visualization with bounding box

🏗 Project Structure
Marine_Object_Classification77/
│
├── app/
│   ├── static/uploads/
│   ├── templates/index.html
│   └── app.py
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── mobilenetv2_marine_classifier.pth
│   ├── efficientnetb0_marine_classifier.pth
│   └── customcnn_marine_classifier.pth
│
├── train_models.py
├── evaluate_models.py
├── plot_results.py
├── requirements.txt
└── README.md

🚀 Installation & Setup
1️⃣ Clone Repository
git clone <your-github-repo-link>
cd Marine_Object_Classification77

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🧪 Train Models
python train_models.py


Trained models will be saved in the models/ folder.

📊 Evaluate Performance
python evaluate_models.py


Generates:

Accuracy

Inference time

Model size (parameters)

📈 Generate Performance Graphs
python plot_results.py


Creates:

accuracy_comparison.png

inference_time_comparison.png

🌐 Run Web Application
cd app
python app.py


Open in browser:

http://127.0.0.1:5000/

📊 Sample Results
Model	Accuracy (%)	Inference Time (ms)	Parameters (M)
MobileNetV2	86.0	19.5	2.23
EfficientNet-B0	85.0	24.9	4.01
Custom CNN	47.0	6.3	0.02
📌 Key Outcomes

Transfer learning significantly improves accuracy

Lightweight models enable faster inference

Model comparison improves transparency

Suitable for real-time and edge deployment

🔮 Future Enhancements

Real-time video classification

True object detection (YOLO/SSD)

More marine species

Edge device deployment

🧾 Technologies Used

Python

PyTorch

Torchvision

Flask

OpenCV

NumPy

HTML/CSS

📬 Contact

For questions or collaborations:

Suma Anjali
Final Year B.Tech Project