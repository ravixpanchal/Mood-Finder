import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(y, sr):
    """
    Creates and returns a matplotlib figure showing the audio waveform.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    # Simple, clear waveform using librosa display tool
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6, color="royalblue")
    ax.set(title="Audio Waveform")
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    """
    Creates and returns a matplotlib figure showing the audio spectrogram.
    """
    # Convert waveform signal to a Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # Convert amplitude to decibels for visualization 
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Spectrogram")
    plt.tight_layout()
    return fig

def plot_mfcc(y, sr):
    """
    Creates and returns a matplotlib figure showing the Mel-Frequency Cepstral Coefficients (MFCCs).
    """
    # Extract MFCCs from the audio signal
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax)
    ax.set(title="Mel-Frequency Cepstral Coefficients (MFCC)")
    plt.tight_layout()
    return fig

def extract_zcr(y):
    """
    Extracts the Zero Crossing Rate (ZCR) features and returns its mean value.
    """
    # Calculate the zero-crossing rate of the audio time series
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Return the mean ZCR across all frames for simplicity
    return np.mean(zcr)
