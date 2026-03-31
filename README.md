# Emotion Detection from Speech

## What is our project?
"Emotion Detection from Speech" (also known as MoodFinder/VoiceEmo) is an interactive, AI-powered web application that analyzes human speech to detect the underlying emotion of the speaker. By uploading a `.wav` audio file, users can visualize acoustical properties of the speech—such as its Waveform, Spectrogram, and Mel-Frequency Cepstral Coefficients (MFCCs)—and receive an AI-generated prediction of the emotional state (e.g., Happy, Sad, Angry, Fearful, etc.) along with a confidence score.

## Why did we select this project?
We chose this project to build a complete, end-to-end application that demonstrates the powerful intersection of Audio Signal Processing (ASP) and deep learning. It serves as an excellent portfolio piece that showcases various skills:
- **Audio Data Handling:** Processing and manipulating raw audio data.
- **Data Visualization:** Translating complex mathematical signal data into readable visual representations.
- **AI/ML Integration:** Seamlessly connecting to state-of-the-art transformer models via cloud APIs.
- **UI/UX Design:** Creating a polished, dark-themed, and responsive web interface that is highly intuitive for end users.

## Which technologies we used
- **Python:** The core programming language used for backend logic and data handling.
- **Streamlit:** The primary web framework utilized to rapidly build the interactive UI.
- **HTML/CSS/JS:** Embedded custom frontend code to create a premium, responsive dark-mode hero section and layout.
- **Librosa & Soundfile:** Powerful Python libraries used for loading, resampling, and extracting features from audio signals.
- **Matplotlib:** Used for rendering high-fidelity visualization charts.
- **Hugging Face Hub API:** Leveraged to communicate with a pre-trained deep learning model (`superb/hubert-large-superb-er` or similar) for highly accurate emotion classification without needing heavy local compute resources.

## How to run
1. **Clone or Download** the project to your local machine.
2. **Setup a Virtual Environment** (recommended) and ensure you have Python 3.8+ installed.
3. **Install Dependencies:** Open your terminal, navigate to the project directory, and run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `librosa` might require `ffmpeg` installed on your system to function properly).*
4. **Start the Application:** Run the Streamlit server using:
   ```bash
   streamlit run app.py
   ```

## How to use
1. **Open your browser** and navigate to the local URL provided by Streamlit (typically `http://localhost:8501`).
2. **Provide your API Token:** In the application's sidebar, enter your free [Hugging Face API Key](https://huggingface.co/settings/tokens) (a 'read' token is sufficient).
3. **Upload Audio:** Use the file uploader in the main application area to select and upload a `.wav` speech file.
4. **Explore Visualizations:** Once uploaded, the app will automatically process the audio and display detailed ASP charts (Waveform, Spectrogram, MFCC, and ZCR).
5. **Predict Emotion:** Scroll down and click the **"Predict Emotion"** button. The application will send the audio to the Hugging Face model and display the recognized emotion along with the confidence score.

---
If you have any issue, feel free to contact ravi.panchal.kaithi@gmail.com
