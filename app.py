import streamlit as st
import whisper
import os
import datetime
import json
from gtts import gTTS
from mt_module.models.huggingface_model import Translator

# Setup
st.set_page_config(page_title="StreamLingo Demo", layout="centered")

st.title("🎧 StreamLingo | English → Chinese Voice Translator")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    with open("temp_input.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio("temp_input.wav", format="audio/wav")
    st.success("✅ Audio uploaded")

    if st.button("▶️ Transcribe, Translate & Generate TTS"):
        with st.spinner("Running Whisper transcription..."):
            try:
                model = whisper.load_model("large-v3")
                result = model.transcribe("temp_input.wav")
                transcript = result["text"]
                st.text_area("📝 Transcript", transcript)
            except Exception as e:
                st.error(f"Whisper failed: {e}")
                st.stop()

        with st.spinner("Translating with NLLB..."):
            try:
                translator = Translator()
                translation = translator.translate(transcript, src_lang="en", tgt_lang="zh")
                st.text_area("🌐 Translated (Chinese)", translation)
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.stop()

        with st.spinner("Generating TTS..."):
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                tts_path = f"tts_output_{timestamp}.mp3"

                tts = gTTS(text=translation, lang="zh")
                tts.save(tts_path)
                st.audio(tts_path, format="audio/mp3")

                # Save transcript and translation to JSON
                json_data = {
                    "transcript": transcript,
                    "translation": translation
                }
                with open(f"result_{timestamp}.json", "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                st.success("✅ JSON saved")
            except Exception as e:
                st.error(f"TTS failed: {e}")
