from stt import transcribe_audio
from mt_module.models.huggingface_model import Translator

audio_path = r"C:\Users\Madhu\OneDrive\Desktop\Project02\sample.wav"


print("🔊 Transcribing...")
text = transcribe_audio(audio_path)
print("Transcript:", text)

print("\n🌐 Translating...")
translator = Translator()
translated = translator.translate(text, src_lang="en", tgt_lang="zh")
print("Translation:", translated)
