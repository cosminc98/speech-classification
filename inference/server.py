import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_on_audio():
    request.files['audio'].save("tmp")
    # print("Posted file: {}".format(request.files['file']))
    # file = request.files['file']
    # return ""
    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4242, debug=True)
