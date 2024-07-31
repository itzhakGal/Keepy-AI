import librosa
import numpy as np
import os
import pickle
from pydub import AudioSegment
import tempfile


def convert_to_wav(file_name):
    audio = AudioSegment.from_file(file_name, format="m4a")
    temp_wav = tempfile.mktemp(suffix=".wav")
    audio.export(temp_wav, format="wav")
    return temp_wav


def extract_features(file_name):
    try:
        if file_name.endswith('.m4a'):
            file_name = convert_to_wav(file_name)

        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file {file_name}: {e}")
        return None


def load_data(data_path):
    features = []
    labels = []
    label_map = {
        '301 - Crying baby': 1,
        '901 - Silence': 0,
        '902 - Noise': 2,
        '903 - Baby laugh': 3
    }

    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return np.array(features), np.array(labels)

    for dir_name in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir_name)
        if os.path.isdir(dir_path) and dir_name in label_map:
            print(f"Loading data from {dir_path}")
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.wav') or file_name.endswith('.m4a'):
                    file_path = os.path.join(dir_path, file_name)
                    print(f"Extracting features from {file_path}")
                    data = extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(label_map[dir_name])
    return np.array(features), np.array(labels)


data_path = os.path.abspath(
    'C:/Users/itzha/Desktop/baby_cry_detection-master/data')  # change this to your actual data path
print(f"Data path is: {data_path}")
features, labels = load_data(data_path)

print(f"Loaded {len(features)} features and {len(labels)} labels")

with open('../model/features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('../model/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
