import asyncio
import numpy as np
import onnxruntime as ort
import librosa  # Make sure librosa is installed

onnx_path = "models/asr_model_optimized.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# Dummy vocabulary (replace with actual vocab if needed)
vocab = [chr(i) for i in range(97, 123)] + [" "]

def greedy_decoder(logits):
    pred_ids = np.argmax(logits, axis=-1)[0]
    tokens = []
    prev = -1
    for idx in pred_ids:
        if idx != prev and idx != 0:  # skip blank token
            if idx < len(vocab):
                tokens.append(vocab[idx])
            else:
                tokens.append("?")
        prev = idx
    return "".join(tokens).strip()


def load_audio(file_path, target_sr=16000):
    """Loads audio from file and returns waveform and sample rate."""
    audio_array, sr = librosa.load(file_path, sr=target_sr)
    return audio_array, sr

def transcribe_audio(audio_array, sample_rate=16000):
    if isinstance(audio_array, str):
        audio_array, sample_rate = load_audio(audio_array, target_sr=sample_rate)

    if not isinstance(audio_array, np.ndarray):
        raise TypeError(f"Expected audio_array to be a NumPy array, got {type(audio_array)}")

    # For raw waveform audio
    input_tensor = np.expand_dims(audio_array.astype(np.float32), axis=0)  # [1, T]
    input_tensor = np.expand_dims(input_tensor, axis=1)                    # [1, 1, T]

    input_length = np.array([input_tensor.shape[-1]], dtype=np.int64)

    inputs = {
        session.get_inputs()[0].name: input_tensor,
        session.get_inputs()[1].name: input_length
    }

    outputs = session.run(None, inputs)
    logits = outputs[0]
    return greedy_decoder(logits)

async def transcribe_audio_async(audio_array, sample_rate=16000):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_audio, audio_array, sample_rate)
