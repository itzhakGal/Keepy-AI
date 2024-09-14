from flask import Flask, render_template, request, jsonify
import os
import subprocess
import time
import threading
import json
import random
import string

app = Flask(__name__)

# Path to the Python script
SCRIPT_PATH = 'main_handler.py'
STATUS_FILE = 'status.txt'


# Function to generate a random password
def generate_random_password(length=8):
    # Use only letters and digits
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))


# Function to check if the script is running
def is_script_running():
    try:
        # Use tasklist to check if the script is running on Windows
        result = subprocess.run(['tasklist'], stdout=subprocess.PIPE, text=True)
        return SCRIPT_PATH.lower() in result.stdout.lower()
    except Exception as e:
        print(f"Error checking script status: {e}")
        return False


# Function to monitor script output
def monitor_script_output():
    global STATUS_FILE
    try:
        process = subprocess.Popen(['python', SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        status_written = False
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())  # Debug print to see the output in the console
                if "Listening..." in output:
                    if not status_written:
                        with open(STATUS_FILE, 'w') as status_file:
                            status_file.write('System is running!\nListening...')
                        status_written = True
    except Exception as e:
        with open(STATUS_FILE, 'w') as status_file:
            status_file.write(f'Error monitoring script output: {e}')


@app.route('/start_script', methods=['POST'])
def start_script():
    try:
        print(f"Starting script: {SCRIPT_PATH}")  # Debug print

        # Write "Trying to start..." to the status file
        with open(STATUS_FILE, 'w') as status_file:
            status_file.write('Trying to start...')

        # Start the script and monitor its output
        subprocess.Popen(['python', SCRIPT_PATH])
        monitoring_thread = threading.Thread(target=monitor_script_output)
        monitoring_thread.start()

        return jsonify({'message': 'Starting "main_handler"...'})
    except Exception as e:
        print(f"Error starting script: {e}")  # Debug print
        return jsonify({'message': f'Error starting script: {e}'}), 500


@app.route('/status')
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as status_file:
            status_message = status_file.read().strip()
    else:
        status_message = 'System isn\'t running!'
    return jsonify({'status_message': status_message})


@app.route('/')
def index():
    # Determine the greeting based on the time of day
    current_hour = int(time.strftime('%H'))
    if 5 <= current_hour < 12:
        greeting = "Good Morning!"
    elif 12 <= current_hour < 17:
        greeting = "Good Afternoon!"
    else:
        greeting = "Good Night!"

    return render_template('index.html', greeting=greeting, status_message='System isn\'t running!')


@app.route('/register', methods=['POST'])
def register():
    try:
        # Get the data from the request
        data = request.json

        # Generate a random password
        password = generate_random_password()

        # Define the file path where the data will be saved
        file_path = 'registrations.json'

        # Check if the file already exists and read the existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                registrations = json.load(file)
        else:
            registrations = []

        # Append the new registration data with the password
        data['password'] = password
        registrations.append(data)

        # Save the updated registrations to the file
        with open(file_path, 'w') as file:
            json.dump(registrations, file, indent=4)

        return jsonify({'message': 'Registration successful!', 'password': password})
    except Exception as e:
        return jsonify({'message': f'Error processing registration: {e}'}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        # Get the login data from the request
        login_data = request.json

        # Load the existing registrations from the JSON file
        file_path = 'registrations.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                registrations = json.load(file)
        else:
            return jsonify({'message': 'No registrations found.', 'success': False})

        # Check if the provided gardenName and password match any entry
        for registration in registrations:
            if (registration['gardenName'] == login_data['gardenName'] and
                registration['password'] == login_data['password']):
                return jsonify({'message': 'Login successful!', 'success': True})

        return jsonify({'message': 'Invalid Kindergarten Name or Password.', 'success': False})
    except Exception as e:
        print(f"Error during login: {e}")  # Debug print
        return jsonify({'message': f'Error during login: {e}', 'success': False}), 500


if __name__ == '__main__':
    # Ensure the status and events files do not exist before starting
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)

    app.run(debug=True)