# 🎙️ Emotion Detection from Speech

> Decode the emotional state hidden within acoustic patterns — with real-time visualizations and multi-format export.

---

## 📖 About the Project

**Emotion Detection from Speech** is a Streamlit-based web application that analyzes uploaded audio files and classifies the speaker's emotional state using acoustic feature extraction and a rule-based classification engine.

No machine learning model training is required — the system uses carefully tuned acoustic thresholds derived from speech psychology literature to produce reliable emotion predictions across a wide range of microphones and recording conditions.

### What it does

- Accepts audio files in **10+ formats** (WAV, MP3, MP4, OGG, FLAC, AAC, M4A, WMA, AIFF, OPUS)
- Visualizes the audio signal as a **Waveform**, **Spectrogram**, and **MFCC heatmap**
- Extracts acoustic features: ZCR, RMS Energy, Spectral Centroid, Spectral Rolloff, Tempo, MFCCs
- Classifies the detected emotion into: **Happy, Sad, Angry, Fearful, Calm, Neutral**
- Shows a **confidence score** and a full probability breakdown for all emotions
- Exports the full report as **PDF**, **Excel (.xlsx)**, or **Word (.docx)**
- Supports **Dark / Light theme** toggle

---

## 🗂️ Project Structure

```
emotion-detection/
│
├── app.py               # Main Streamlit application (UI + logic)
├── emotion_engine.py    # Rule-based emotion classifier
├── utils.py             # Feature extraction & visualization helpers
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ How It Works

### Feature Extraction (`utils.py`)

| Feature | Description |
|---|---|
| **ZCR** | Zero Crossing Rate — how often the signal crosses zero |
| **RMS** | Root Mean Square Energy — loudness proxy |
| **Spectral Centroid** | Brightness / pitch-related centre of mass |
| **Spectral Rolloff** | Frequency below which 85% of energy is contained |
| **Tempo** | Estimated speech rhythm in BPM |
| **MFCCs** | 13 Mel-Frequency Cepstral Coefficients |

### Emotion Classification (`emotion_engine.py`)

Each feature is normalized to a `[-1, +1]` range using configurable thresholds. A weighted linear combination of these normalized scores is computed for each of the six emotions. The raw scores are then **softmax-normalized** into a probability distribution.

```
Angry   →  high ZCR + high RMS + high Rolloff + fast Tempo
Happy   →  high Centroid + high RMS + fast Tempo + high ZCR
Sad     →  low ZCR + low RMS + low Centroid + slow Tempo
Fearful →  high ZCR + high Centroid + high Rolloff + fast Tempo
Calm    →  low ZCR + low RMS + slow Tempo + low Rolloff
Neutral →  moderate values across all features
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- `ffmpeg` installed and on your system PATH (required for non-WAV formats)

#### Install ffmpeg

| Platform | Command |
|---|---|
| **macOS** | `brew install ffmpeg` |
| **Ubuntu/Debian** | `sudo apt install ffmpeg` |
| **Windows** | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH |

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Running Locally

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `librosa` | Audio loading & feature extraction |
| `matplotlib` | Waveform, spectrogram & MFCC plots |
| `numpy` | Numerical operations |
| `soundfile` | Audio file I/O backend |
| `pydub` | MP3/MP4/OGG/FLAC conversion (requires ffmpeg) |
| `reportlab` | PDF report generation |
| `openpyxl` | Excel (.xlsx) report generation |
| `python-docx` | Word (.docx) report generation |

---

## 🌐 Deployment

### Option 1 — Streamlit Community Cloud (Recommended — Free)

1. Push your project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** → select your repo, branch, and set **Main file path** to `app.py`
5. Click **"Deploy"** — your app will be live in ~2 minutes

> **Note:** Streamlit Community Cloud does **not** include `ffmpeg` by default.  
> To enable MP3/MP4 support, add a `packages.txt` file to your repo root:
> ```
> ffmpeg
> ```

---

### Option 2 — Render (Free tier available)

1. Push your project to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Set the following:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**

> Render's free tier spins down after inactivity — first load may be slow.

---

### Option 3 — Railway

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub**
2. Select your repo
3. Add environment variable: `PORT=8501`
4. Set start command: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
5. Deploy

---

### Option 4 — Hugging Face Spaces (Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create new Space**
2. Choose **Streamlit** as the SDK
3. Upload your files or connect your GitHub repo
4. Hugging Face will auto-detect `requirements.txt` and deploy

> Add `ffmpeg` to a `packages.txt` file in the root for full audio format support.

---

### Option 5 — Docker (Self-hosted / VPS)

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t emotion-detection .
docker run -p 8501:8501 emotion-detection
```

Access at **http://localhost:8501**

---

## 🎨 Features

| Feature | Details |
|---|---|
| 🌗 Theme Toggle | Switch between Dark and Light mode |
| 🌊 Waveform | Amplitude vs time plot |
| 📊 Spectrogram | STFT frequency heatmap |
| 🎨 MFCC | 13-coefficient cepstral visualization |
| 🤖 Emotion Classification | 6 emotion categories with confidence |
| 📄 PDF Export | Styled report via ReportLab |
| 📊 Excel Export | Two-sheet workbook via OpenPyXL |
| 📝 Word Export | Formatted .docx via python-docx |
| 🎵 Multi-format Audio | WAV, MP3, MP4, OGG, FLAC, AAC, M4A, WMA, AIFF, OPUS |

---

## 🐛 Troubleshooting

**MP3/MP4 not loading**
- Ensure `ffmpeg` is installed and accessible from your terminal (`ffmpeg -version`)
- On Windows, add the ffmpeg `bin/` folder to your system PATH

**Missing export buttons**
- Install the optional export packages: `pip install reportlab openpyxl python-docx`

**librosa errors**
- Ensure `soundfile` is installed: `pip install soundfile`
- On Linux you may also need: `sudo apt install libsndfile1`

---

## 📄 License

This project is intended for educational and research purposes.

---

<p align="center">Made with ❤️ by Team. All Rights Reserved © 2026</p>
