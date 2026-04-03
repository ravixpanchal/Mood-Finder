"""
emotion_engine.py
-----------------
Rule-based emotion classifier using acoustic features extracted by utils.py.

Features used
─────────────
  zcr       – Zero Crossing Rate
                High  → Angry / Fearful (rapid, irregular speech)
                Low   → Calm / Sad (slow, steady speech)

  rms       – Root Mean Square Energy (loudness proxy)
                High  → Angry / Happy (energetic speech)
                Low   → Sad / Calm (quiet speech)

  centroid  – Spectral Centroid (brightness / pitch-related)
                High  → Happy / Fearful (bright, high-pitched)
                Low   → Sad / Neutral (dull, low-pitched)

  rolloff   – Spectral Rolloff (presence of high-frequency content)
                High  → Angry / Fearful
                Low   → Calm / Sad

  tempo     – Speech rhythm / rate
                Fast  → Happy / Angry
                Slow  → Sad / Calm / Neutral

  mfcc[1]   – 2nd MFCC coefficient (timbral warmth)
                Negative → Sad / Calm
                Positive → Happy / Angry

Each emotion gets a raw score from weighted evidence; scores are then
softmax-normalised into a probability distribution.
"""

import math
import numpy as np


# ── Thresholds tuned on typical RAVDESS-range feature values ───────────────
# These are deliberately wide so they work on a variety of microphones and
# recording conditions without a trained model.

_ZCR_LOW   = 0.03   # below → slow/calm speech
_ZCR_HIGH  = 0.08   # above → rapid/agitated speech

_RMS_LOW   = 0.02   # below → quiet / subdued
_RMS_HIGH  = 0.06   # above → loud / energetic

_CEN_LOW   = 1400   # Hz  below → dull / low-pitched
_CEN_HIGH  = 2200   # Hz  above → bright / high-pitched

_ROL_LOW   = 2800   # Hz  below → muffled
_ROL_HIGH  = 4500   # Hz  above → sharp

_TEMPO_LOW  = 80    # BPM below → slow speech
_TEMPO_HIGH = 120   # BPM above → fast speech


def _score(value, low, high):
    """
    Returns (+1, 0, -1) depending on whether value is above, between or below
    the low/high thresholds.  Used as a fuzzy evidence weight.
    """
    if value >= high:
        return 1.0
    if value <= low:
        return -1.0
    # linear interpolation inside the band
    mid = (low + high) / 2.0
    return (value - mid) / (high - mid)


def classify_emotion(features: dict):
    """
    Classify emotion from the feature dict returned by utils.extract_features().

    Parameters
    ----------
    features : dict
        Keys: zcr, rms, centroid, rolloff, tempo, mfcc

    Returns
    -------
    emotion    : str   — e.g. "Happy"
    confidence : float — 0-100
    scores     : dict  — {emotion_name: percentage}  (all sum to 100)
    """

    z = _score(features["zcr"],      _ZCR_LOW,   _ZCR_HIGH)
    r = _score(features["rms"],      _RMS_LOW,   _RMS_HIGH)
    c = _score(features["centroid"], _CEN_LOW,   _CEN_HIGH)
    o = _score(features["rolloff"],  _ROL_LOW,   _ROL_HIGH)
    t = _score(features["tempo"],    _TEMPO_LOW, _TEMPO_HIGH)
    m = float(np.sign(features["mfcc"][1]))   # sign of 2nd MFCC

    # ── Raw evidence rules ─────────────────────────────────────────────────
    # Each value is a linear combination of the normalised feature scores.
    # Positive weights = feature reinforces this emotion.
    # Signs are chosen based on acoustic psychology literature.

    raw = {
        "Angry":   1.5*z + 1.5*r + 0.8*o + 0.6*t + 0.4*m,
        "Happy":   1.2*c + 1.0*r + 0.8*t + 0.6*z + 0.4*m,
        "Sad":    -1.2*z - 1.2*r - 1.0*c - 0.6*t - 0.4*m,
        "Fearful": 1.0*z + 0.8*c - 0.6*r + 0.8*o + 0.4*t,
        "Calm":   -1.0*z - 0.8*r - 0.6*t - 0.4*o,
        "Neutral":  0.2 - 0.3*abs(z) - 0.3*abs(r) - 0.2*abs(c),
    }

    # ── Softmax normalisation ──────────────────────────────────────────────
    keys   = list(raw.keys())
    values = np.array([raw[k] for k in keys], dtype=float)

    # Shift so min is 0 before softmax to avoid huge negatives dominating
    values -= values.min()
    exp_v   = np.exp(values)          # temperature=1 gives reasonable spread
    probs   = exp_v / exp_v.sum()

    scores = {k: round(float(p) * 100, 1) for k, p in zip(keys, probs)}

    best_emotion    = max(scores, key=scores.get)
    best_confidence = scores[best_emotion]

    return best_emotion, best_confidence, scores
