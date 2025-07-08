from stt import transcribe_audio
from mt_module.models.huggingface_model import Translator

# File to process
audio_path = "sample.wav"  # Replace with your actual file name

# Step 1: Transcribe
print("🔊 Transcribing...")
transcript = transcribe_audio(audio_path)
print("Transcript:\n", transcript)

# Step 2: Translate
print("\n🌐 Translating to Chinese...")
translator = Translator()
translation = translator.translate(transcript, src_lang="en", tgt_lang="zh")
print("Translation:\n", translation)
