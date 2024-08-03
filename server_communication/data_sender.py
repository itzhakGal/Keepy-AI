import requests
from config import Config


class DataSender:
    def __init__(self):
        self.url = Config.SERVER_URL
        self.headers = {'Content-Type': 'application/json'}

    def send_json_data(self, data):
        print("Sending JSON data:", data)
        response = requests.post(self.url + "process-data", json=data, headers=self.headers)
        print("HTTP Response for data:", response.status_code)

    def send_audio_file(self, audio_file_path):
        print("Sending audio file:", audio_file_path)
        # Uncomment the next three lines to actually send the audio file
        files = {'file': open(audio_file_path, 'rb')}
        response = requests.post(self.url + "upload-audio", files=files)
        print("HTTP Response for audio:", response.status_code)
