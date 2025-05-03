import torch
import torch.nn as nn
import argparse
import os

# --- Raw Signal Model definition ---
class RawSignalCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=(3,1,5), padding=(1,0,2)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,1,3), padding=(1,0,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript")
    parser.add_argument('--model', type=str, default="best_rawsignal_cnn.pth", 
                        help="Path to model .pth file")
    parser.add_argument('--output', type=str, default=None, 
                        help="Output path for TorchScript model (default: <model_name>_jit.pt)")
    parser.add_argument('--num-classes', type=int, default=27, 
                        help="Number of output classes")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.model)[0]
        args.output = f"{base_name}_jit.pt"
    
    print(f"Loading model from {args.model}...")
    
    # Create and load model
    model = RawSignalCNN(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 6, 5, 1, 91)  # Batch, Features, Sensors, 1, Window
    
    # Convert to TorchScript via tracing
    print("Converting model to TorchScript...")
    traced_script_module = torch.jit.trace(model, example_input)
    
    # Save the TorchScript model
    traced_script_module.save(args.output)
    print(f"TorchScript model saved to {args.output}")
    
    # Verify the model works
    print("Verifying model...")
    loaded_model = torch.jit.load(args.output)
    test_output = loaded_model(example_input)
    print(f"Test output shape: {test_output.shape}")
    print("Conversion successful!")

if __name__ == "__main__":
    main()
