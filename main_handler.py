import os
import uuid
import wave

import librosa
import pyaudio
import threading
import numpy as np
from collections import deque
from datetime import datetime

from config import Config
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
        self.cnn_crying_model = load_model(Config.CNN_MODEL_PATH)
        self.sentence_classifier = SentenceClassifier()
        self.kindergarten_name = 'tali'
        self.data_sender = DataSender()

        self.buffer_duration = 180  # 3 minutes
        self.sample_rate = 44100  # Assuming 44.1 kHz
        self.buffer_size = self.buffer_duration * self.sample_rate
        self.audio_buffer = deque(maxlen=self.buffer_size)

        self.is_processing_event = False
        self.microphone_lock = threading.Lock()
        self.event_audio_data = None  # Initialize this attribute

    def generate_unique_id(self):
        return str(uuid.uuid4())

    def main(self):
        audio_format = pyaudio.paInt16
        channels = 1
        chunk = 2048
        rate = self.sample_rate

        p = pyaudio.PyAudio()
        stream = p.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

        try:
            while True:
                with self.microphone_lock:
                    text, audio_data = self.audio_processor.listen_for_speech()

                # Convert audio_data to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.update_buffer(audio_array)

                waveform = self.audio_processor.audio_to_waveform(audio_data, sample_rate=rate)

                if not self.is_processing_event:
                    # Capture the event's audio data
                    self.event_audio_data = audio_array
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

    def update_buffer(self, audio_array):
        """Add new audio data to the buffer."""
        self.audio_buffer.extend(audio_array)

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

    def create_and_send_event(self, event_type, input_data, detected_data=None):
        event_id = self.generate_unique_id()
        event_data = {
            "id": event_id,
            "event": event_type,
            "timestamp": self.get_current_time(),
            "kindergarten_name": self.kindergarten_name
        }
        if event_type == "crying_detected":
            intensity = self.calculate_crying_intensity(input_data, sample_rate=self.sample_rate)
            duration = self.calculate_crying_duration(input_data, sample_rate=self.sample_rate)
            event_data["intensity"] = intensity
            event_data["duration"] = duration

        if event_type == "curse_word_detected" and detected_data:
            event_data["word"] = detected_data
        elif event_type == "inappropriate_sentence_detected":
            event_data["sentence"] = input_data

        threading.Thread(target=self.process_event, args=(event_data, event_type, event_id)).start()

    def process_event(self, event_data, event_name, event_id):
        self.is_processing_event = True
        print(
            f"Event: {event_data['event']}, Timestamp: {event_data['timestamp']}, ID: {event_data['id']}, Kindergarten: {event_data['kindergarten_name']}")
        self.data_sender.send_json_data(event_data)

        # Calculate the number of samples for 2 minutes before and 1 minute after
        pre_event_samples = 2 * 60 * self.sample_rate
        post_event_samples = 1 * 60 * self.sample_rate

        # Extract the pre-event audio data from the buffer
        with self.microphone_lock:
            print(f"Current buffer length: {len(self.audio_buffer)} samples")

            # If buffer is smaller than required samples, pad with silence
            if len(self.audio_buffer) < pre_event_samples:
                pre_event_audio = np.zeros(pre_event_samples, dtype=np.int16)
                pre_event_audio[-len(self.audio_buffer):] = np.array(self.audio_buffer)
                print(
                    f"Buffer too short. Padding pre-event audio with silence. Pre-event length: {len(pre_event_audio)} samples")
            else:
                pre_event_audio = np.array(self.audio_buffer)[-pre_event_samples:]
                print(f"Extracted pre-event audio: {len(pre_event_audio)} samples")

        # Start capturing post-event audio
        post_event_audio = []
        print("Capturing 1 minute of post-event audio...")

        # Keep capturing for 1 minute
        stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                                        frames_per_buffer=1024)

        try:
            for _ in range(int(post_event_samples / 1024)):
                audio_data = stream.read(1024)
                post_event_audio.extend(np.frombuffer(audio_data, dtype=np.int16))
        finally:
            stream.stop_stream()
            stream.close()

        post_event_audio = np.array(post_event_audio, dtype=np.int16)
        print(f"Captured post-event audio: {len(post_event_audio)} samples")

        # Combine the pre-event, event, and post-event audio
        combined_audio = np.concatenate([pre_event_audio, self.event_audio_data, post_event_audio])
        print(f"Combined audio length: {len(combined_audio)} samples")

        # Convert combined_audio to bytes
        combined_audio_bytes = combined_audio.tobytes()

        # Save and send the audio file
        server_audio_filepath = self.save_audio_file(combined_audio_bytes, self.sample_rate, 1, 'send_audio_server',
                                                     filename=f'{event_id}.wav')
        print(f"Audio file saved: {server_audio_filepath}")
        self.data_sender.send_audio_file(server_audio_filepath)
        print(f"Audio file sent to server: {server_audio_filepath}")

        self.is_processing_event = False

    def save_audio_file(self, audio_data, sample_rate, channels, output_path, filename):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filepath = os.path.join(output_path, filename)

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

        return filepath

    def cleanup(self, stream, p):
        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_current_time(self):
        now = datetime.now()
        return f"{now.day}/{now.month}/{now.year} {now.strftime('%H:%M:%S')}"

    def calculate_crying_intensity(self, waveform, sample_rate=16000):
        # Compute the Root Mean Square (RMS) energy for the waveform
        rms_energy = librosa.feature.rms(y=waveform)[0]

        # Normalize the RMS energy to a 1-10 scale for intensity
        intensity = np.mean(rms_energy) * 10  # This is a simplified scaling
        intensity = min(max(intensity, 1), 10)  # Clip values to be between 1 and 10

        return intensity

    def calculate_crying_duration(self, waveform, sample_rate=16000):
        # Apply a simple threshold to determine where crying occurs in the waveform
        threshold = 0.02  # Adjust this threshold as necessary
        non_silent_intervals = librosa.effects.split(y=waveform, top_db=20)

        # Calculate total duration of crying in seconds
        total_crying_duration = 0
        for start, end in non_silent_intervals:
            segment_duration = (end - start) / sample_rate
            total_crying_duration += segment_duration

        return int(total_crying_duration)


if __name__ == "__main__":
    handler = MainHandler()
    handler.main()
