import requests
from config import Config


class DataSender:
    def __init__(self):
        self.url = Config.SERVER_URL
        self.headers = {'Content-Type': 'application/json'}

    def send_data(self, data):
        response = requests.post(self.url, json=data, headers=self.headers)
        print("HTTP Response:", response.status_code)