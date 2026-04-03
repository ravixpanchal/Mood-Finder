import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(y, sr):
    """Waveform: amplitude vs time."""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.6, color="royalblue")
    ax.set(title="Audio Waveform", xlabel="Time (s)", ylabel="Amplitude")
    plt.tight_layout()
    return fig


def plot_spectrogram(y, sr):
    """Short-Time Fourier Transform spectrogram in dB."""
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="hz", sr=sr, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Spectrogram")
    plt.tight_layout()
    return fig


def plot_mfcc(y, sr):
    """13 Mel-Frequency Cepstral Coefficients over time."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax)
    ax.set(title="MFCC (Mel-Frequency Cepstral Coefficients)")
    plt.tight_layout()
    return fig


def extract_features(y, sr):
    """
    Extract all acoustic features needed for emotion classification.

    Returns a dict with:
        zcr       – Zero Crossing Rate (mean)
        rms       – Root Mean Square energy (mean)
        centroid  – Spectral Centroid in Hz (mean)
        rolloff   – Spectral Rolloff in Hz (mean)
        tempo     – Estimated tempo in BPM
        mfcc      – np.ndarray of shape (13,) — mean of each MFCC coefficient
    """
    # Zero Crossing Rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # RMS Energy
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # Spectral Centroid
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Spectral Rolloff (85% energy threshold)
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # MFCCs — mean of each of the 13 coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    return {
        "zcr":      zcr,
        "rms":      rms,
        "centroid": centroid,
        "rolloff":  rolloff,
        "tempo":    tempo,
        "mfcc":     mfcc,
    }
