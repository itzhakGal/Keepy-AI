import pyaudio
import wave
import os
from datetime import datetime
import shutil
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
        self.create_audio_folder()

    def create_audio_folder(self):
        if not os.path.exists('audioSounds'):
            os.makedirs('audioSounds')

    def save_audio_file(self, audio_data, sample_rate, channels, output_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'audio_{timestamp}.wav'
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

                # Save the audio data to a file
                self.save_audio_file(audio_data, rate, channels, 'audioSounds')

                waveform = self.audio_processor.audio_to_waveform(audio_data, sample_rate=rate)

                self.handle_crying_detection(waveform)
                self.handle_curse_detection(text)
                self.handle_sentence_classification(text)

        except KeyboardInterrupt:
            print("Exiting...")
            self.cleanup(stream, p)

    def handle_crying_detection(self, waveform):
        if self.crying_detector.detect_crying(waveform):
            event_data = {
                "event": "crying_detected",
                "timestamp": self.get_current_time()
            }
            print(f"Event: {event_data['event']}, Timestamp: {event_data['timestamp']}")
            # self.data_sender.send_data(event_data)

    def handle_curse_detection(self, text):
        curses = self.curse_detector.detect_curses(text)
        if curses:
            event_data = {
                "event": "curse_word_detected",
                "word": curses,
                "timestamp": self.get_current_time()
            }
            print(f"Event: {event_data['event']}, Word: {event_data['word']}, Timestamp: {event_data['timestamp']}")
            # self.data_sender.send_data(event_data)

    def handle_sentence_classification(self, text):
        if text.strip():  # Check if text is not empty or only whitespace
            label = self.sentence_classifier.classify_sentence(text)
            if label == 'inappropriate':
                event_data = {
                    "event": "inappropriate_sentence_detected",
                    "sentence": text,
                    "timestamp": self.get_current_time()
                }
                print(
                    f"Event: {event_data['event']}, Sentence: {event_data['sentence']}, Timestamp: {event_data['timestamp']}")
                # self.data_sender.send_data(event_data)

    def cleanup(self, stream, p):
        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_current_time(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    handler = MainHandler()
    handler.main()
