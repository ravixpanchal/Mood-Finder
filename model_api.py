import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# The Hugging Face model URL for speech emotion recognition
# This model uses the superb dataset and outputs emotions like angr, hap, neu, sad.
API_URL = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

def detect_emotion(audio_bytes, api_key):
    """
    Sends the audio file to the Hugging Face API and returns the emotion prediction.

    Args:
        audio_bytes (bytes): The raw bytes of the uploaded .wav file.
        api_key (str): The Hugging Face User Access Token.

    Returns:
        dict: A dictionary containing 'emotion' on success or 'error' on failure.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "audio/wav"
    }
    
    # Set up session with retry strategy for resilient network requests
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    MAX_ATTEMPTS = 3
    for attempt in range(MAX_ATTEMPTS):
        try:
            # POST request to send the audio bytes directly to the Inference API
            response = session.post(API_URL, headers=headers, data=audio_bytes, timeout=60)
            
            # Check if the model is currently loading (Hugging Face specific behavior)
            if response.status_code == 503 and "estimated_time" in response.text:
                time.sleep(5) # Wait 5 seconds and retry
                continue
                
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                # Find the prediction with the highest score
                best_prediction = max(result, key=lambda x: x['score'])
                
                return {
                    "success": True,
                    "emotion": best_prediction['label'],
                    "score": best_prediction['score']
                }
            else:
                return {
                    "success": False,
                    "error": f"API Error {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.ChunkedEncodingError as e:
            # specifically catch IncompleteRead (which inherits from ChunkedEncodingError)
            if attempt == MAX_ATTEMPTS - 1:
                return {"success": False, "error": f"Connection broke while receiving the response. The audio file might be too large or the API server dropped the connection. Try a shorter audio file."}
            time.sleep(2)
            
        except requests.exceptions.ReadTimeout:
            if attempt == MAX_ATTEMPTS - 1:
                return {"success": False, "error": "The API request timed out. The model might be overloaded. Please try again later."}
            time.sleep(2)
            
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                return {
                    "success": False,
                    "error": f"Connection Error: {str(e)}\nMake sure you have a stable internet connection and try again."
                }
            time.sleep(2)
