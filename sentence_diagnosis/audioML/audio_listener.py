import speech_recognition as sr

def listen_for_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text.lower(), audio.get_raw_data()
    except sr.UnknownValueError:
        return "", audio.get_raw_data()
