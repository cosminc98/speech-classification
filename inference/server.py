import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import requests
import tempfile
import librosa
import soundfile

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_on_audio():
    tmp_dirname = tempfile.TemporaryDirectory()

    audio_fpath = os.path.join(tmp_dirname.name, "sample.webm")
    request.files['audio'].save(audio_fpath)

    audio, sr = librosa.load(audio_fpath, sr=16000)
    wav_fpath = os.path.join(tmp_dirname.name, "sample.wav")
    soundfile.write(file=wav_fpath, data=audio, samplerate=sr, subtype="PCM_16")

    tmp_dirname.cleanup()

    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4242, debug=True)
