import numpy as np
import librosa

def preprocess_audio(audio, sample_rate):
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    audio = np.array(audio).astype(np.float32)
    audio = np.expand_dims(audio, axis=0)
    return audio

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]
