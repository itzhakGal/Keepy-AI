# 🤖 Keepy - AI 🤖

### 📝 Overview
The **Keepy - AI** repository contains the machine learning models and audio processing algorithms responsible for detecting unusual sounds, such as crying, shouting, and other anomalies, in kindergartens. The AI component processes audio data and communicates with the server for further handling of notifications and storage.

---

## 📁 Structure

- **cry_diagnosis/**: Contains scripts and models used for detecting crying in the audio data.
- **curse_diagnosis/**: Contains scripts and models used for detecting profanity in the audio data.
- **positive_feedback/**: Handles positive feedback classification based on audio events.
- **sentence_diagnosis/**: Contains scripts and models used for detecting inappropriate sentences.
- **server_communication/**: Handles communication between the AI and the server.
- **templates/**: Contains template files used for various AI-related tasks.

- **app.py**: Main application script for running the AI module.
- **audio_processor.py**: Handles audio processing tasks like feature extraction.
- **config.py**: Configuration file for setting up paths and parameters for the AI system.
- **events.txt**: Logs events related to detected audio anomalies.
- **main_handler.py**: Main handler for orchestrating the AI processes and sending results to the server.
- **registrations.json**: Stores registration information for different audio event types.

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

