from nemo.collections.asr.models import EncDecCTCModel
import torch
import os

def load_model():
    # Load pretrained ASR model from NeMo hub
    #model = EncDecCTCModel.from_pretrained(model_name="stt_hi_conformer_ctc_medium")
    model = EncDecCTCModel.restore_from("models/stt_hi_conformer_ctc_medium.nemo")
    print("Model loaded successfully.")
    model.eval()
    return model

def export_onnx_model(model, output_path="models/asr_modell.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create dummy input tensor (batch=1, 16000 audio samples)
    input_example = torch.randn(1, 16000, dtype=torch.float32)

    # Export ONNX without input_sample arg (fixed per NeMo version)
    #model.export(output=output_path, input=input_example)
    model.export(output="models/asr_modell.onnx")
    # Save vocabulary for decoding
    vocab_path = os.path.join(os.path.dirname(output_path), "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for c in model.decoder.vocabulary:
            f.write(c + "\n")
    print(f"ONNX model and vocab exported to {os.path.dirname(output_path)}")

if __name__ == "__main__":
    model = load_model()
    export_onnx_model(model)


result = model.transcribe(["hindi_test_6sec_16k.wav"])
print("Transcription:", result[0])

