import torch
import onnx
import os
import sys
import requests
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

# Configuration
# The checkpoint uses 'vit_l' (ViT-Large) backbone with 1024 dimensions
# We will use 'cpsam' in the output filename to match the model name.
MODEL_TYPE = "vit_l" 
OUTPUT_MODEL_PATH = "cellpose_cpsam.onnx"
CHECKPOINT_PATH = "cpsam.pth"
CELLPOSE_SAM_URL = "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam"

def download_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Downloading Cellpose-SAM checkpoint from {CELLPOSE_SAM_URL}...")
        response = requests.get(CELLPOSE_SAM_URL, stream=True)
        response.raise_for_status()
        with open(CHECKPOINT_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Checkpoint found at {CHECKPOINT_PATH}")

def export_cellpose_sam_full():
    print("Detected Cellpose-SAM architecture (ViT Backbone + Cellpose Head).")
    print("Exporting as end-to-end Cellpose model...")
    
    # We need to instantiate the actual Cellpose model class
    # We can rely on the 'cellpose' library to load it for us
    from cellpose.vit_sam import Transformer
    
    # Initialize model (vit_l architecture)
    # The checkpoint uses ViT-Large backbone (1024 dimensions)
    # Parameters: nout=3 (flows+cellprob)
    model = Transformer(backbone=MODEL_TYPE, nout=3)
    
    # Load weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Clean state dict if needed (remove 'module.' prefix if DDP was used)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Dummy input
    # The model has a fixed positional embedding of 32x32
    # With patch size 8, this means input must be 256x256 (32*8=256)
    # Dynamic axes are not supported due to fixed pos_embed dimensions
    dummy_input = torch.randn(1, 3, 256, 256)
    
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_MODEL_PATH,
        export_params=True,
        opset_version=18,  # Updated to 18 as recommended by warning
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "style"], # forward returns x1, style
        # Note: No dynamic_axes - model requires fixed 256x256 input due to fixed pos_embed
    )
    print(f"Export complete: {OUTPUT_MODEL_PATH}")

def main():
    download_checkpoint()
    
    # Based on investigation, Cellpose-SAM is an end-to-end segmentation model, not an interactive one.
    # So we export the full model.
    export_cellpose_sam_full()

if __name__ == "__main__":
    main()
