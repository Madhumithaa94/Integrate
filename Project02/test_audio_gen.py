import numpy as np
import scipy.io.wavfile as wavfile

# Generate a test sine wave (1 second, 440Hz tone)
samplerate = 16000  # 16kHz sample rate
duration = 1        # in seconds
frequency = 440     # A4 note

t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save as .wav
wavfile.write("test.wav", samplerate, audio.astype(np.float32))

print("✅ test.wav generated successfully.")
