import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def check_model(path: str):
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"ONNX model {path} is valid.")
    except Exception as e:
        print(f"Model check failed: {e}")
        raise


def optimize_onnx_model(input_path: str, output_path: str):
    model = onnx.load(input_path)
    onnx.save(model, output_path)  # Simply saves or rewrites the model
    print(f"✅ Non-quantized optimized model saved to {output_path}")

if __name__ == "__main__":
    input_path = "models/asr_model.onnx"
    output_path = "models/asr_model_optimized.onnx"
    optimize_onnx_model(input_path, output_path)

