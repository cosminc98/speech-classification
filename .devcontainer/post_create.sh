python3.9 -m pip install flask Flask-cors librosa pandas
sudo apt -y install libsndfile-dev ffmpeg
echo "VITE_INFERENCE_API_URL=\"https://${CODESPACE_NAME}-4242.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN}\"" > website/.env