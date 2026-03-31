import streamlit as st
import librosa
import soundfile as sf
import io

# Import our helper modules
from utils import plot_waveform, plot_spectrogram, plot_mfcc, extract_zcr
from model_api import detect_emotion

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Emotion Detection from Speech",
    page_icon="🎙️",
    layout="wide"
)

# --- CUSTOM THEME & UI HEADER ---
st.markdown("""
<style>
/* CSS to create the dark grid background across the whole app */
.stApp {
    background-color: #0b0914;
    background-image: 
        linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: white;
}

/* Hide the default generic Streamlit top padding so our hero looks better */
.block-container {
    padding-top: 2rem;
}

/* Hero Section Container */
.hero-container {
    text-align: center;
    padding-top: 30px;
    padding-bottom: 40px;
}

/* Top Purple Badge */
.top-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 50px;
    background-color: rgba(34, 197, 94, 0.08); /* slight green tint */
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    margin-bottom: 25px;
}

/* Main Hero Typography */
.main-title {
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 20px;
    color: #ffffff;
    font-family: "Inter", sans-serif;
}

/* Vibrant Blue/Cyan Highlight */
.highlight {
    color: #00d2ff;
}

/* Subtitle paragraph */
.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Flexbox row for bottom responsive badges */
.badges-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-top: 40px;
}

.feature-badge {
    padding: 8px 18px;
    border-radius: 50px;
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Responsive Scaling for Mobile Users */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.8rem;
    }
    .subtitle {
        font-size: 0.95rem;
        padding: 0 15px;
    }
}
</style>
<div class="hero-container"><div class="top-badge">🟢 EMOTION AI • SPEECH ANALYSIS</div><div class="main-title">Hear what words<br><span class="highlight">can't always say</span></div><div class="subtitle">Upload speech audio and let our ML model decode the emotional state hidden within acoustic patterns — with real-time ASP visualizations and confidence scores.</div><div class="badges-row"><div class="feature-badge">🌊 Waveform Analysis</div><div class="feature-badge">📊 FFT Spectrum</div><div class="feature-badge">🤖 ML Classification</div><div class="feature-badge">🎨 Mel-Spectrogram</div></div></div>
""", unsafe_allow_html=True)

# --- API KEY SETUP ---
# Look for API key in secrets first. If not found, ask the user in the UI.
hf_api_key = "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"
try:
    if "HF_API_KEY" in st.secrets:
        hf_api_key = st.secrets["HF_API_KEY"]
except FileNotFoundError:
    pass

if not hf_api_key:
    hf_api_key = st.sidebar.text_input("Enter Hugging Face API Key", type="password")
    st.sidebar.markdown(
        "Don't have an API key? Get it completely free at [Hugging Face](https://huggingface.co/settings/tokens)."
    )

if not hf_api_key:
    st.warning("⚠️ Please provide a Hugging Face API key to use the Emotion Prediction feature.")

st.markdown("---")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload an audio file (.wav format)", type=["wav"])

if uploaded_file is not None:
    # --- 1. PLAY THE AUDIO ---
    st.audio(uploaded_file, format="audio/wav")
    
    # --- 2. LOAD AUDIO USING LIBROSA ---
    try:
        # Load the uploaded raw bytes using librosa
        y, sr = librosa.load(io.BytesIO(uploaded_file.getvalue()), sr=None)
        
        st.success("✅ Audio loaded successfully!")
        
        # --- 3. FEATURE EXTRACTION & VISUALIZATION ---
        st.header("🔍 Audio Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Waveform")
            st.markdown("A waveform shows how the amplitude of the audio signal changes over time.")
            fig_wave = plot_waveform(y, sr)
            st.pyplot(fig_wave)
        
        with col2:
            st.subheader("Spectrogram")
            st.markdown("A spectrogram visualizes the frequencies of a given signal as it varies with time.")
            fig_spec = plot_spectrogram(y, sr)
            st.pyplot(fig_spec)

        st.markdown("---")
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            st.subheader("MFCC (Mel Frequency Cepstral Coefficients)")
            st.markdown("MFCCs represent the short-term power spectrum of a sound. They are highly useful for speech processing.")
            fig_mfcc = plot_mfcc(y, sr)
            st.pyplot(fig_mfcc)
            
        with col4:
            st.subheader("Zero Crossing Rate (ZCR)")
            st.markdown("The rate at which the signal goes from positive to zero to negative or vice versa.")
            zcr_val = extract_zcr(y)
            st.metric(label="Mean ZCR", value=f"{zcr_val:.4f}")
            
        st.markdown("---")
            
        # --- 4. EMOTION PREDICTION ---
        # st.header("🧠 Emotion Prediction")
        
        # if st.button("Predict Emotion", type="primary"):
        #     if not hf_api_key:
        #         st.error("Please provide your Hugging Face API key on the left sidebar to proceed.")
        #     else:
        #         with st.spinner("Analyzing the speech emotion using Hugging Face model..."):
        #             target_sr = 16000
                    
        #             # FIX: Use librosa.to_mono() instead of y[0] for correct stereo → mono conversion
        #             y_mono = librosa.to_mono(y) if y.ndim > 1 else y
                    
        #             y_api = librosa.resample(y_mono, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y_mono
                    
        #             max_len = target_sr * 10
        #             if len(y_api) > max_len:
        #                 y_api = y_api[:max_len]
        #                 st.info("ℹ️ Your audio was longer than 10 seconds. We truncated it for predicting to improve API reliability and speed.")
                        
        #             # Write to an in-memory byte buffer
        #             buffer = io.BytesIO()
        #             sf.write(buffer, y_api, target_sr, format='WAV', subtype='PCM_16')
        #             buffer.seek(0)
        #             api_audio_bytes = buffer.read()
                    
        #             # API call from our model_api.py helper
        #             result = detect_emotion(api_audio_bytes, hf_api_key)
                    
        #             if result.get("success"):
        #                 # FIX: Updated emotion_map to match new model's labels
        #                 emotion_map = {
        #                     "angry":     ("Angry",     "😡"),
        #                     "calm":      ("Calm",      "😌"),
        #                     "disgust":   ("Disgust",   "🤢"),
        #                     "fearful":   ("Fearful",   "😨"),
        #                     "happy":     ("Happy",     "😄"),
        #                     "neutral":   ("Neutral",   "😐"),
        #                     "sad":       ("Sad",       "😢"),
        #                     "surprised": ("Surprised", "😲"),
        #                 }
                        
        #                 raw_label = result['emotion'].lower()  # normalize to lowercase for safe matching
        #                 formatted_label, emoji = emotion_map.get(raw_label, (raw_label.capitalize(), "🤔"))
        #                 confidence = result['score'] * 100
                        
        #                 st.success("Prediction Complete!")
        #                 st.subheader(f"Predicted Emotion: {emoji} **{formatted_label}**")
        #                 st.progress(result['score'], text=f"Confidence: {confidence:.2f}%")
                        
        #             else:
        #                 st.error(f"Failed to predict emotion: {result.get('error')}")

    except Exception as e:
        st.error(f"Error processing the audio file: {e}")
        st.error("Please make sure you uploaded a valid .wav file.")