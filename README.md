# 🤖 Keepy - AI 🤖

### 📝 Overview
The **Keepy - AI** repository contains the machine learning models and audio processing algorithms responsible for detecting unusual sounds, such as crying, shouting, and other anomalies, in kindergartens. The AI component processes audio data and communicates with the server for further handling of notifications and storage.

---

## 📁 Structure

- **models/**: Contains the pre-trained models used for sound classification.
- **scripts/**: Audio processing scripts that handle feature extraction, model inference, and interaction with the server.

---

## 🔧 Installation

### Requirements:
- **Python 3.x**.
- **TensorFlow** and other dependencies (listed in `requirements.txt`).

---

## 📚 Usage

1. The AI system receives raw audio data from kindergartens.
2. Extracts features from the audio and classifies it using pre-trained models to detect anomalies such as crying, profanity, and inappropriate sentences.
3. Sends the classification results to the server for notification and storage.
4. Optionally, retrains the model with new data to improve accuracy.

---

## 💡 Features

- **Sound Classification**: Detects various sound patterns using machine learning, focusing on identifying abnormal sounds like crying, profanity, and inappropriate sentences.
- **Model Training**: Supports retraining the models with new data to improve detection accuracy.
- **Feature Extraction**: Processes raw audio data to extract relevant features used by the AI models.
- **Server Integration**: Sends detection results to the **Keepy Server** for further handling and notification.

