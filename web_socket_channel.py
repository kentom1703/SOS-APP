import sounddevice as sd
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import os
import tempfile
import soundfile as sf
import asyncio
import websockets

REFERENCE_DIR = "reference_audios"
THRESHOLD = 0.3
DURATION = 5  # Duration of recording in seconds
SAMPLE_RATE = 22050

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def load_reference_fingerprints():
    fingerprints = {}
    for filename in os.listdir(REFERENCE_DIR):
        if filename.endswith((".wav", ".mp3")):
            path = os.path.join(REFERENCE_DIR, filename)
            fingerprints[filename] = extract_mfcc(path)
    return fingerprints

async def recognize_live_audio(websocket, reference_fingerprints):
    print("Listening for accident sound... Speak or play a sound now.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, recording, SAMPLE_RATE)
        test_mfcc = extract_mfcc(tmpfile.name)

    for ref_name, ref_mfcc in reference_fingerprints.items():
        similarity = 1 - cosine(test_mfcc, ref_mfcc)
        print(f"Compared with {ref_name}: Similarity = {similarity:.2f}")
        if similarity > (1 - THRESHOLD):
            print("Accident Detected!")
            await websocket.send("accident_detected")
            return True
    print("No accident audio detected.")
    return False

async def main():
    reference_fingerprints = load_reference_fingerprints()
    async with websockets.serve(
        lambda websocket, path: recognize_live_audio(websocket, reference_fingerprints),
        "localhost",
        8765
    ):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())