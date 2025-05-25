from nemo.collections.asr.models import EncDecCTCModel
import torch
import os

os.makedirs("models", exist_ok=True)

# Load model from local .nemo file (use restore_from, NOT from_pretrained)
#model = EncDecCTCModel.restore_from("stt_hi_conformer_ctc_medium.nemo")
model = EncDecCTCModel.restore_from("models/stt_hi_conformer_ctc_medium.nemo")

model.eval()

# Create a dummy input tensor matching model input: batch_size=1, audio_len=16000 (1 sec of 16kHz audio)
# The model expects shape (batch, time) float tensor
input_sample = torch.randn(1, 16000, dtype=torch.float32)

# Export to ONNX
model.export(output="models/asr_model.onnx")

# Export vocabulary file (one token per line)
with open("models/vocab.txt", "w", encoding="utf-8") as f:
    for c in model.decoder.vocabulary:
        f.write(c + "\n")

print("Model and vocabulary exported successfully.")

import nemo.collections.asr as nemo_asr

# Load pretrained Hindi ASR model


# Transcribe the generated 6-sec audio
result = model.transcribe(["hindi_test_6sec_16k.wav"])
print("Transcription:", result[0])
