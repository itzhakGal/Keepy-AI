# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import librosa

from cry_diagnosis.machine_learning_model.baby_cry_detection.pc_methods import Reader
from cry_diagnosis.machine_learning_model.baby_cry_detection.pc_methods.feature_engineer import FeatureEngineer


def load_model(model_path):
    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)
    return model


def predict_crying(model, audio_file):
    # Read and process the audio file
    reader = Reader(audio_file)
    data, sample_rate = reader.read_audio_file()

    # Feature engineering
    feature_engineer = FeatureEngineer()
    features, _ = feature_engineer.feature_engineer(audio_data=data)

    # Make prediction
    prediction = model.predict(features)
    return prediction


def main():
    model_path = '{}/../../../baby_cry_detection-master/output/model/model.pkl'.format(
        os.path.dirname(os.path.abspath(__file__)))
    audio_file = 'audio_20240727_134244.wav'  # Replace with your audio file path

    # Load the trained model
    model = load_model(model_path)

    # Predict if the audio contains crying
    prediction = predict_crying(model, audio_file)

    # Display the result
    if prediction[0] == '301 - Crying baby':
        print("The audio clip contains baby crying.")
    else:
        print("The audio clip does not contain baby crying.")


if __name__ == '__main__':
    main()
