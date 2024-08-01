import pyaudio
import wave
import os
from datetime import datetime, timedelta
import threading
from audio_processor import AudioProcessor
from cry_diagnosis.yamnet_model.crying_detector import CryingDetector
from curse_diagnosis.curse_detector import CurseDetector
from sentence_diagnosis.sentence_classifier import SentenceClassifier
from server_communication.data_sender import DataSender


class MainHandler:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.curse_detector = CurseDetector()
        self.crying_detector = CryingDetector()
        self.sentence_classifier = SentenceClassifier()
        self.data_sender = DataSender()
        self.create_audio_folders()
        self.buffer = []
        self.buffer_duration = 30  # seconds

    def create_audio_folders(self):
        if not os.path.exists('audioSounds'):
            os.makedirs('audioSounds')
        if not os.path.exists('send_audio_server'):
            os.makedirs('send_audio_server')

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
                text, audio_data = self.audio_processor.listen_for_speech()

                # Add audio data to buffer
                self.buffer.append((audio_data, datetime.now()))
                self.buffer = [(data, ts) for data, ts in self.buffer if
                               (datetime.now() - ts).seconds <= self.buffer_duration]

                waveform = self.audio_processor.audio_to_waveform(audio_data, sample_rate=rate)

                self.handle_crying_detection(waveform)
                self.handle_curse_detection(text)
                self.handle_sentence_classification(text)

        except KeyboardInterrupt:
            print("Exiting...")
            self.cleanup(stream, p)

    def handle_crying_detection(self, waveform):
        if self.crying_detector.detect_crying(waveform):
            self.send_event_data("crying_detected")

    def handle_curse_detection(self, text):
        curses = self.curse_detector.detect_curses(text)
        if curses:
            self.send_event_data("curse_word_detected", {"word": curses})

    def handle_sentence_classification(self, text):
        if text.strip():  # Check if text is not empty or only whitespace
            label = self.sentence_classifier.classify_sentence(text)
            if label == 'inappropriate':
                self.send_event_data("inappropriate_sentence_detected", {"sentence": text})

    def send_event_data(self, event_type, additional_data=None):
        start_time = datetime.now() - timedelta(seconds=10)
        end_time = datetime.now() + timedelta(seconds=20)
        audio_clip = self.get_audio_clip(start_time, end_time)
        filename = f'{event_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
        filepath = self.save_audio_file(audio_clip, 44100, 1, 'send_audio_server', filename)

        event_data = {
            "event": event_type,
            "timestamp": self.get_current_time()
        }
        if additional_data:
            event_data.update(additional_data)

        self.data_sender.send_json_data(event_data)
        self.data_sender.send_audio_file(filepath)

    def get_audio_clip(self, start_time, end_time):
        audio_clip = b''
        for data, ts in self.buffer:
            if start_time <= ts <= end_time:
                audio_clip += data
        return audio_clip

    def cleanup(self, stream, p):
        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_current_time(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    handler = MainHandler()
    handler.main()
