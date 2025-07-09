import os
import json
import datetime
import whisper

from mt_module.models.huggingface_model import Translator
from gtts import gTTS
from playsound import playsound

# 📁 Ask user for audio input
audio_path = input("📁 Enter full path to your audio file:\n").strip()

if not os.path.isfile(audio_path):
    print("❌ Error: File does not exist.")
    exit()

# 🔊 Whisper Transcription
print("\n🔊 Transcribing with Whisper large-v3...\n")
try:
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    print("✅ Transcription complete.")
    print("📄 Transcript:\n", transcript)
except Exception as e:
    print("❌ Whisper transcription failed:")
    print(e)
    exit()

# 🌐 Translate using NLLB
print("\n🌐 Translating to Chinese...\n")
try:
    translator = Translator()
    translated = translator.translate(transcript, src_lang="en", tgt_lang="zh")
    print("✅ Translation complete.")
    print("🈶 Translated Text:\n", translated)
except Exception as e:
    print("❌ Translation failed:")
    print(e)
    exit()

# 🔊 TTS using gTTS
print("\n🎤 Generating TTS audio with gTTS...\n")
try:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tts_output_path = f"tts_output_{timestamp}.mp3"

    tts = gTTS(text=translated, lang='zh')
    tts.save(tts_output_path)
    print(f"✅ TTS audio saved as: {tts_output_path}")

    # Optionally play it
    playsound(tts_output_path)
except Exception as e:
    print("❌ TTS generation failed:")
    print(e)
    exit()

# 💾 Save to JSON
print("\n💾 Saving result JSON...\n")
try:
    result_data = {
        "filename": os.path.basename(audio_path),
        "transcript": transcript,
        "translation": translated
    }
    json_path = f"result_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Results saved at: {json_path}")
except Exception as e:
    print("❌ Saving result failed:")
    print(e)
