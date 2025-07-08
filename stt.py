import whisper
import os

model = whisper.load_model("large-v3")

def transcribe_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    
    result = model.transcribe(audio_path)
    return result["text"]
