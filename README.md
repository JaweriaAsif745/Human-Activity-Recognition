# 🎬 Human Action Recognition (CNN + LSTM)

**Real-time Human Action Recognition from Video & Webcam using Deep Learning**

This project implements a **real-time action recognition system** using a **CNN + LSTM architecture** in **PyTorch**. The system can detect human actions from **video files** or **live webcam streams** and display the **top 3 predictions with confidence scores**, highlighting the top action in green.

---

## 🌟 Features

* **Video & Webcam Input**: Recognize actions from uploaded videos or live webcam.
* **Top-2 Predictions**: Shows the two most probable actions; top prediction highlighted on video frame.
* **Temporal Modeling**: LSTM processes sequences of 16 frames to capture motion over time.
* **Checkpointing**: Save and resume model training safely.
* **Interactive Web App**: Built with **Streamlit** for an easy-to-use interface.

---
📚 Dataset: UCF101
Link: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition

This project uses the UCF101 action recognition dataset
 hosted on Kaggle.

UCF101 contains 13,320 video clips across 101 human action classes, collected from YouTube. 
crcv.ucf.edu
+2
crcv.ucf.edu
+2

The dataset is very challenging due to:

Diverse actions — 101 categories, including sports, body motions, human-object interactions, musical instruments, and more. 
crcv.ucf.edu
+1

Real-world variability — large variations in camera motion, background clutter, viewpoint, object scale, lighting, and occlusion. 
crcv.ucf.edu
+1

Group structure — videos are divided into 25 groups, each containing several clips. 
crcv.ucf.edu

Official train/test splits are provided by the authors to ensure consistent evaluation. 
crcv.ucf.edu

Using UCF101 helps train and evaluate video models that generalize well to “in-the-wild” human actions.

---

## 🖥 Demo

Run the app locally with:

```bash
streamlit run app.py
```

* Upload a video (mp4, avi) or start your webcam.
* See real-time predictions displayed on the video frame and below the video.
* Top prediction appears in **green**, while others are shown in standard text.

---

## 📷 Screenshots

### **Home Screen**

!<img width="917" height="408" alt="home" src="https://github.com/user-attachments/assets/cd451015-d5f2-45fb-8672-d305ed744aba" />


### **Results Screen**

<img width="1920" height="1931" alt="Action-Recognition-App" src="https://github.com/user-attachments/assets/4665c2ea-1b3a-44b0-8c92-a426cccf58fe" />

---

## 🧰 Tech Stack

* **Python 3.8+**
* **PyTorch** (CNN + LSTM)
* **torchvision**
* **OpenCV** for video processing
* **Streamlit** for web app deployment
* **NumPy & Pillow** for data handling and transforms

---

## 📂 Project Structure

```
├── app.py                        # Streamlit web application
├── src/
│   ├── model.py                  # CNN + LSTM architecture
│   ├── predict.py                # Video inference and prediction
├── models/
│   └── action_recognition_model.pth  # Pretrained weights
├── screenshots/
│   ├── home.png           # Home page screenshot
│   └── result.png        # Prediction results screenshot
├── README.md
└── requirements.txt               # Required Python packages
```

---

## 🏗 Installation

1. **Clone the repository**

```bash
git clone https://github.com/JaweriaAsif745/Human-Activity-Recognition.git
cd human-action-recognition
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## ⚡ Usage

1. **Upload Video**: Choose any mp4, avi, or mov file.
2. **Live Webcam**: Start your webcam for real-time action recognition.
3. **View Predictions**:

   * Top prediction is displayed **on the video frame in green**.
   * Top 2 predictions are displayed below the frame.

---

## 📊 Model Overview

* **CNN Encoder**: ResNet18 pretrained on ImageNet extracts spatial features from frames.
* **LSTM Layer**: Processes sequences of frame features to capture temporal dynamics.
* **Classifier**: Outputs probabilities for each action class.
* **Checkpointing**: Save model state during training to resume or fine-tune later.

---

## 🏆 Results

* Real-time prediction of human actions on uploaded videos and webcam.
* Works with **UCF101 dataset** and custom videos.
* Shows the **top-2 predicted actions** with probabilities.

---

## 🚀 Future Improvements

* Optimize for **GPU deployment** for faster inference.
* Smooth live webcam streaming for higher FPS.
* Expand to **more datasets and action classes**.
* Integrate **multi-user web app interface**.

---

## 📜 License

MIT License © Jaweria Asif Khan
