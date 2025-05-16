import torch

# Load your PyTorch model
model = classifier1bertfinetuned.pt

# Convert the model to ONNX
input_example = torch.randn(1, 3, 224, 224)  # Provide an example input shape
onnx_path = "model.onnx"  # Path to save the ONNX model
torch.onnx.export(model, input_example, onnx_path)