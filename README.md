# Configuration

If not running from GitHub Codespaces, you'll need to manually create the "website/.env" file with the following environment variable to allow the web interface to know where to make inference requests:
```bash
VITE_INFERENCE_API_URL="https://localhost:4242"
```

## (OPTIONAL) Add the large version of the CNN models
- github codespaces: right click on "inference/models/\<task\>/cnn" and click on "upload" then add the "model_large.pt" files
- local project: move the "model_large.pt" files to inference "inference/models/\<task\>/cnn"

Then change the key "MODEL_SIZE" from the "inference/.env" environment file to "large".

# Web Interface

Start the web interface:
```bash
cd ./website
npm install
npm run dev
```

# Inference Server

Start the inference server:
```bash
cd ./inference
python3.9 server.py
```

Due to [this issue](https://github.com/orgs/community/discussions/15351), you must change the visibility of port 4242 to "Public".
