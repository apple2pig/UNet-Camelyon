"""
Quick Start Example for Optimized Inference
Replace paths with your actual file paths before running
"""

from inference_optimized import OptimizedInference
import torch

# Check available device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize optimized inference engine
print("\nInitializing optimized inference engine...")
engine = OptimizedInference(
    model_path='UNet_17.pth',        # Your trained model path
    device=device,
    use_fp16=True,                   # Enable half precision (2-3x faster)
    use_jit=True,                    # Enable JIT compilation (20% faster)
    batch_size=12,                   # Adjust based on your GPU memory
    num_workers=4                    # Number of threads for data loading
)

# ============================================================
# Example 1: Process a single WSI
# ============================================================
print("\n" + "="*60)
print("Example 1: Process Whole Slide Image")
print("="*60)

wsi_path = '/Camelyon16/test_040.tif'  # Replace with your WSI path
output_path = 'wsi_result.png'

try:
    stats = engine.process_wsi_batched(
        wsi_path=wsi_path,
        patch_size=512,
        output_path=output_path,
        overlap=0  # Set to 32 or 64 for smoother boundaries
    )

    print(f"\n✓ WSI processing complete!")
    print(f"  Total patches: {stats['total_patches']}")
    print(f"  Processing speed: {stats['patches_per_sec']:.2f} patches/sec")
    print(f"  Total time: {stats['elapsed_time']:.2f} seconds")

except FileNotFoundError:
    print(f"⚠ WSI file not found: {wsi_path}")
    print(f"  Please update the wsi_path variable with your actual file path")

# ============================================================
# Example 2: Process directory of patches
# ============================================================
print("\n" + "="*60)
print("Example 2: Process Patches Directory")
print("="*60)

input_dir = 'patch_path/'           # Replace with your input directory
output_dir = 'output_path/'         # Replace with your output directory

try:
    engine.process_patches_directory(
        input_dir=input_dir,
        output_dir=output_dir
    )

    print(f"\n✓ Batch processing complete!")
    print(f"  Results saved to: {output_dir}")

except FileNotFoundError:
    print(f"⚠ Input directory not found: {input_dir}")
    print(f"  Please update the input_dir variable with your actual directory path")

# ============================================================
# Example 3: Export model to ONNX (for deployment)
# ============================================================
print("\n" + "="*60)
print("Example 3: Export to ONNX Format")
print("="*60)

try:
    engine.export_to_onnx(
        output_path='unet_optimized.onnx',
        input_size=(1, 3, 512, 512)
    )

    print(f"\n✓ ONNX export complete!")
    print(f"  You can now use this model with:")
    print(f"    - ONNX Runtime (CPU/GPU)")
    print(f"    - TensorRT (NVIDIA)")
    print(f"    - OpenVINO (Intel)")
    print(f"    - C++/Java/JavaScript applications")

except Exception as e:
    print(f"⚠ ONNX export failed: {e}")

# ============================================================
# Tips for optimizing performance
# ============================================================
print("\n" + "="*60)
print("Performance Tuning Tips")
print("="*60)
print("""
1. Adjust batch_size based on GPU memory:
   - 6GB GPU:  batch_size=4-6
   - 8GB GPU:  batch_size=8-10
   - 12GB GPU: batch_size=12-16
   - 24GB GPU: batch_size=16-24

2. If you get CUDA Out of Memory errors:
   - Reduce batch_size
   - Set use_fp16=False
   - Reduce patch_size from 512 to 256

3. For CPU inference:
   - Set device='cpu'
   - Set use_fp16=False
   - Use smaller batch_size (2-4)

4. For smoother WSI results:
   - Set overlap=32 or overlap=64
   - Note: This increases processing time by ~15-30%

5. Run benchmark to measure speedup:
   python compare_inference_speed.py
""")

print("\n✓ All examples complete! Check the output files.")
