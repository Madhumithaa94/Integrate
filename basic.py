import whisper
import subprocess
import shutil
import os

# Step 1: File paths
original_path = r"C:\Users\Madhu\OneDrive\Desktop\Project02\sample.wav"
copied_path = "copied_sample.wav"
converted_path = "clean_sample.wav"

# Step 2: Try copying the file
try:
    shutil.copy2(original_path, copied_path)
    print("✅ File copied successfully.")
except Exception as e:
    print("❌ Failed to copy file:")
    print(e)
    exit()

# Step 3: Convert the copied file using ffmpeg
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # Overwrite without asking
    "-i", copied_path,
    "-ac", "1",          # Mono
    "-ar", "16000",      # 16kHz
    converted_path
]

try:
    print("🔄 Converting audio with ffmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)
    print("✅ Audio converted.")
except subprocess.CalledProcessError as e:
    print("❌ ffmpeg failed:")
    print(e)
    exit()

# Step 4: Transcribe using Whisper
try:
    print("\n🔊 Transcribing with Whisper...\n")
    model = whisper.load_model("large-v3")
    result = model.transcribe(converted_path)

    print("✅ Transcription complete.")
    print("📄 Transcript:\n")
    print(result["text"])
except Exception as e:
    print("❌ Whisper failed:")
    print(e)
