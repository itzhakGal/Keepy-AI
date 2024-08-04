import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
from config import Config

class CryingDetector:
    def __init__(self):
        self.yamnet_model = hub.load(Config.YAMNET_MODEL_HANDLE)
        self.class_names = self.load_class_names(Config.CSV_FILEPATH)

    def load_class_names(self, filepath):
        class_names = []
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                class_names.append(row[2])
        return np.array(class_names)

    def audio_to_waveform(self, audio_data, sample_rate=16000):
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        waveform = audio_np / 32768.0
        return tf.convert_to_tensor(waveform, dtype=tf.float32)

    def detect_crying(self, waveform):
        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        scores_np = scores.numpy()
        crying_sobbing_index = np.where(self.class_names == 'Crying, sobbing')[0][0]
        baby_cry_index = np.where(self.class_names == 'Baby cry, infant cry')[0][0]
        crying_scores = scores_np[:, crying_sobbing_index] + scores_np[:, baby_cry_index]
        return crying_scores.max() > 0.2
