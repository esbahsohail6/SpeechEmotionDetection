from flask import Flask, render_template, request
import numpy as np
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
import uuid
import matplotlib
matplotlib.use('Agg')  # <-- use this before importing pyplot
import matplotlib.pyplot as plt


app = Flask(__name__)
model = load_model("model.h5")

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

def save_waveform(data, sr, filename):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(data, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    path = f'static/{filename}'
    plt.savefig(path)
    plt.close()
    return path

def save_spectrogram(data, sr, filename):
    X = librosa.stft(data)
    X_db = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(X_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.title("Spectrogram")
    plt.tight_layout()
    path = f'static/{filename}'
    plt.savefig(path)
    plt.close()
    return path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['audio']
        if file:
            # Save file
            file_id = str(uuid.uuid4())
            file_path = f'temp_{file_id}.wav'
            file.save(file_path)

            # Load audio
            data, sr = librosa.load(file_path, sr=None)
            duration = len(data) / sr

            # Generate plots
            waveform_path = save_waveform(data, sr, f'waveform_{file_id}.png')
            spectrogram_path = save_spectrogram(data, sr, f'spectrogram_{file_id}.png')

            # Predict emotion
            mfcc_features = extract_mfcc(file_path)
            mfcc_input = np.expand_dims(mfcc_features, axis=0)
            prediction = model.predict(mfcc_input)
            print(f"predivtion:{prediction}")
            emotion_labels = {
                0: "Angry",
                1: "Disgust",
                2: "Fear",
                3: "Happy",
                4: "Neutral",
                5: "Sad",
                6: "Surprise"
            }
            detected_emotion = emotion_labels[np.argmax(prediction)]


            # Clean up audio file
            os.remove(file_path)

            return render_template('result.html',
                       sample_rate=sr,
                       duration=duration,
                       waveform=os.path.basename(waveform_path),
                       spectrogram=os.path.basename(spectrogram_path),
                       emotion=str(detected_emotion))


    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
