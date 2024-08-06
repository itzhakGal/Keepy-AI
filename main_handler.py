import uuid
import librosa
import pyaudio
import wave
import os
from datetime import datetime
import threading
import numpy as np
from audio_processor import AudioProcessor
from cry_diagnosis.yamnet_model.crying_detector import CryingDetector
from curse_diagnosis.curse_detector import CurseDetector
from sentence_diagnosis.sentence_classifier import SentenceClassifier
from server_communication.data_sender import DataSender
from tensorflow.keras.models import load_model


class MainHandler:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.curse_detector = CurseDetector()
        self.yamnet_crying_detector = CryingDetector()
        self.cnn_crying_model = load_model('cry_diagnosis/machine_learning_model/cnn/model/baby_cry_cnn_model.h5')
        self.sentence_classifier = SentenceClassifier()
        self.kindergarten_name = 'tali'
        self.data_sender = DataSender()
        self.previous_audio_data = b""
        self.event_audio_data = b""
        self.after_event_audio_data = b""
        self.is_processing_event = False
        self.microphone_lock = threading.Lock()

    def generate_unique_id(self):
        return str(uuid.uuid4())

    def save_audio_file(self, audio_data, sample_rate, channels, output_path, filename):
        filepath = os.path.join(output_path, filename)

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        return filepath

    def main(self):
        audio_format = pyaudio.paInt16
        channels = 1
        chunk = 2048
        rate = 44100

        p = pyaudio.PyAudio()
        stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        try:
            while True:
                with self.microphone_lock:
                    text, audio_data = self.audio_processor.listen_for_speech()

                self.previous_audio_data = self.event_audio_data
                self.event_audio_data = audio_data

                waveform = self.audio_processor.audio_to_waveform(audio_data, sample_rate=rate)

                if not self.is_processing_event:
                    self.handle_event_detection("crying_detected", self.detect_crying, waveform)
                    self.handle_event_detection("curse_word_detected", self.curse_detector.detect_curses, text)
                    self.handle_event_detection("inappropriate_sentence_detected",
                                                self.sentence_classifier.classify_sentence, text)

        except KeyboardInterrupt:
            print("Exiting...")
            self.cleanup(stream, p)

    def detect_crying(self, waveform):
        yamnet_detected = self.yamnet_crying_detector.detect_crying(waveform)
        cnn_detected = self.detect_crying_with_cnn(waveform.numpy())
        return yamnet_detected or cnn_detected

    def detect_crying_with_cnn(self, audio_data):
        sample_rate = 16000
        features = self.extract_features(audio_data, sample_rate)
        if features is not None:
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=2)
            prediction = self.cnn_crying_model.predict(features, verbose=0)
            return np.argmax(prediction, axis=1)[0] == 1
        return False

    def extract_features(self, audio_data, sample_rate):
        try:
            audio_data = audio_data / np.max(np.abs(audio_data))
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled
        except Exception as e:
            print(f"Error encountered while parsing audio data: {e}")
            return None

    def handle_event_detection(self, event_name, detection_function, input_data):
        if event_name == "crying_detected" and detection_function(input_data):
            self.create_and_send_event(event_name, input_data)

        elif event_name == "curse_word_detected":
            detected_data = detection_function(input_data)
            if detected_data:
                self.create_and_send_event(event_name, input_data, detected_data)

        elif event_name == "inappropriate_sentence_detected" and input_data.strip() and detection_function(
                input_data) == 'inappropriate':
            self.create_and_send_event(event_name, input_data)

    def create_and_send_event(self, event_name, input_data, detected_data=None):
        event_id = self.generate_unique_id()
        event_data = {
            "id": event_id,
            "event": event_name,
            "timestamp": self.get_current_time(),
            "kindergarten_name": self.kindergarten_name
        }

        if event_name == "curse_word_detected" and detected_data:
            event_data["word"] = detected_data
        elif event_name == "inappropriate_sentence_detected":
            event_data["sentence"] = input_data

        threading.Thread(target=self.process_event, args=(event_data, event_name, event_id)).start()

    def process_event(self, event_data, event_name, event_id):
        self.is_processing_event = True
        print(
            f"Event: {event_data['event']}, Timestamp: {event_data['timestamp']}, ID: {event_data['id']}, Kindergarten: {event_data['kindergarten_name']}")
        self.data_sender.send_json_data(event_data)

        with self.microphone_lock:
            self.after_event_audio_data = self.audio_processor.listen_for_speech()[1]

        combined_audio = self.previous_audio_data + self.event_audio_data + self.after_event_audio_data
        server_audio_filepath = self.save_audio_file(combined_audio, 44100, 1, 'send_audio_server',
                                                     filename=f'{event_id}.wav')
        self.data_sender.send_audio_file(server_audio_filepath)
        self.is_processing_event = False

    def cleanup(self, stream, p):
        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_current_time(self):
        now = datetime.now()
        return f"{now.day}/{now.month}/{now.year} {now.strftime('%H:%M:%S')}"


if __name__ == "__main__":
    handler = MainHandler()
    handler.main()
