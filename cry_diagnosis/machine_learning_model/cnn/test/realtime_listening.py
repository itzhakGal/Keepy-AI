import librosa
import numpy as np
import os
import pickle
from pydub import AudioSegment
import tempfile
import noisereduce as nr
import pyaudio
import wave
import tensorflow as tf
from tensorflow.keras.models import load_model
import speech_recognition as sr
from datetime import datetime


def convert_to_wav(file_name):
    audio = AudioSegment.from_file(file_name, format="m4a")
    temp_wav = tempfile.mktemp(suffix=".wav")
    audio.export(temp_wav, format="wav")
    return temp_wav


def extract_features(audio_data, sample_rate):
    try:
        audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate)

        audio_data = librosa.util.normalize(audio_data)

        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing audio data: {e}")
        return None


def save_audio_file(audio_data, sample_rate, channels, output_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'audio_{timestamp}.wav'
    filepath = os.path.join(output_path, filename)
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data)
    wf.close()
    return filepath


def predict_crying(model, audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    if features is not None:
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        prediction = model.predict(features)
        return np.argmax(prediction, axis=1)[0]
    else:
        return None


def listen_for_crying(model, listen_duration=5):
    output_path = '../audioSounds'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for ambient noise

    print("Listening for baby cries...")

    try:
        while True:
            with microphone as source:
                print('Recording...')
                audio = recognizer.record(source, duration=listen_duration)
                audio_data = audio.get_raw_data()

                sample_rate = source.SAMPLE_RATE
                channels = 1  # Assume mono recording

                # Save audio file
                audio_file_path = save_audio_file(audio_data, sample_rate, channels, output_path)
                print(f"Audio file saved to: {audio_file_path}")

                # Convert frames to numpy array
                audio_data_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                audio_data_np = audio_data_np / 32768.0  # Normalize to [-1, 1]

                # Log audio data
                print(f"Audio data (first 10 samples): {audio_data_np[:10]}")
                print(f"Audio max level: {np.max(audio_data_np)}, min level: {np.min(audio_data_np)}")

                # Predict if the audio contains baby crying
                prediction = predict_crying(model, audio_data_np, sample_rate)

                # Display the result if crying is detected
                if prediction == 1:  # Assuming label 1 is for 'Crying baby'
                    print("Baby crying detected!")
                else:
                    print("No baby crying detected.")

    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == '__main__':
    # Load the trained model
    model = load_model('../model/baby_cry_cnn_model.h5')

    listen_for_crying(model, listen_duration=5)
