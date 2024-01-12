import speech_recognition as sr

# Create a recognizer object
r = sr.Recognizer()

# Load the WAV file
with sr.AudioFile('C:/Users/TRUNG QUAN/Downloads/file.wav') as source:
    audio_data = r.record(source)

# Convert speech to text with Vietnamese language
text = r.recognize_google(audio_data, language='vi-VN')
print(text)