import speech_recognition as sr
import numpy as np
import tensorflow as tf


class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen_for_speech(self):
        with self.microphone as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            # No need to manually create sr.AudioData, use audio object directly
            text = self.recognizer.recognize_google(audio)
            return text.lower(), audio.get_raw_data()
        except sr.UnknownValueError:
            return "", audio.get_raw_data()

    def audio_to_waveform(self, audio_data, sample_rate=16000):
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        waveform = audio_np / 32768.0
        return tf.convert_to_tensor(waveform, dtype=tf.float32)