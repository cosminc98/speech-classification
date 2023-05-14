import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import requests
import tempfile
import librosa
import soundfile
import random
import json
from pickle import load
from dotenv import load_dotenv

from svm import extract_features, N_MFCC
from cnn import load_cnn_model, predict_cnn
from utils import AudioFile

# load environment variables from ".env"
load_dotenv()
try: 
    model_size = os.environ["MODEL_SIZE"]
except KeyError:
    model_size = "small"

app = Flask(__name__)
CORS(app)

models = {
    "yesno": {
        "svm": {
            "pca": load(open("models/yesno/svm/pca.pkl", 'rb')),
            "scaler": load(open("models/yesno/svm/scaler.pkl", 'rb')),
            "model": load(open("models/yesno/svm/model.pkl", 'rb')),
            "id_to_label": {
                0: "no",
                1: "yes",
            },
            "n_seconds": 1.0,
        },
        "cnn": {
            "model": load_cnn_model(
                model_fpath=f"models/yesno/cnn/model_{model_size}.pt", 
                n_classes=2,
                use_large_model=(model_size=="large"),
            ),
            "id_to_label": {
                0: "no",
                1: "yes",
            },
        },
    },
    "ser": {
        "svm": {
            "pca": load(open("models/ser/svm/pca.pkl", 'rb')),
            "scaler": load(open("models/ser/svm/scaler.pkl", 'rb')),
            "model": load(open("models/ser/svm/model.pkl", 'rb')),
            "id_to_label": {
                0: "angry",
                1: "happy",
                2: "sad",
            },
            "n_seconds": 2.0,
        },
        "cnn": {
            "model": load_cnn_model(
                model_fpath=f"models/ser/cnn/model_{model_size}.pt", 
                n_classes=3,
                use_large_model=(model_size=="large"),
            ),
            "id_to_label": {
                0: "angry",
                1: "happy",
                2: "sad",
            },
        },
    }
}

@app.route('/predict', methods=['POST'])
def predict_on_audio():
    tmp_dirname = tempfile.TemporaryDirectory()

    audio_fpath = os.path.join(tmp_dirname.name, "sample.webm")
    request.files['audio'].save(audio_fpath)

    config = json.loads(request.files['config'].read().decode('utf-8'))
    task = config['task']
    model = config['model']

    print(f'[INFO] Running prediction for task "{task}" with model "{model}"')

    audio, sr = librosa.load(audio_fpath, sr=16000)
    wav_fpath = os.path.join(tmp_dirname.name, "sample.wav")
    soundfile.write(file=wav_fpath, data=audio, samplerate=sr, subtype="PCM_16")

    task_models = models[task]
    if model == "svm":
        svm_model = task_models["svm"]
        pca = svm_model["pca"]
        scaler = svm_model["scaler"]
        model = svm_model["model"]
        id_to_label = svm_model["id_to_label"]
        n_seconds = svm_model["n_seconds"]

        features_predict, _ = extract_features(
            audio_files=[
                AudioFile(
                    file_path=wav_fpath, 
                    subset="predict", 
                    speaker_id="UNKNOWN", 
                    utterance_id="predict_sample", 
                    label="UNKNOWN"
                )
            ],
            label_to_id=None,
            subset="predict",
            n_mfcc=N_MFCC,
            n_seconds=n_seconds,
        )

        features_predict_scaled = scaler.transform(features_predict)
        features_predict_pca = pca.transform(features_predict_scaled)

        prediction_index = int(model.predict(features_predict_pca)[0])
        prediction_name = id_to_label[prediction_index]

    else:
        cnn_model = task_models["cnn"]
        model = cnn_model["model"]
        id_to_label = cnn_model["id_to_label"]
        prediction_name = predict_cnn(
            audio_fpath=wav_fpath,
            model=model,
            id_to_label=id_to_label
        )

    response = jsonify({'label': prediction_name})

    tmp_dirname.cleanup()

    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4242, debug=True)
