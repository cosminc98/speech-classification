```bash
# start the web interface
cd ./website
npm install
npm run dev
```

Due to [this issue](https://github.com/orgs/community/discussions/15351), you must change the visibility of port 4242 to "Public".

If not running from GitHub Codespaces, you'll need to manually create the "website/.env" file with the following environment variable:
```bash
VITE_INFERENCE_API_URL="https://localhost:4242"
```