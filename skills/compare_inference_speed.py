"""
Side-by-side comparison of original vs optimized inference
Run this script to see the performance improvement
"""

import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_dummy_patch(size=512):
    """Create a dummy patch for testing"""
    return Image.new('RGB', (size, size), color='white')

def original_inference_single(model, patch, device, transforms_fn):
    """Original single-patch inference (from pre_WSI.py)"""
    x = transforms_fn(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0]
    pred = torch.max(pred, dim=0).values
    pred = pred.cpu().numpy()
    return (pred * 255).astype(np.uint8)

def benchmark_original(model_path, device, num_patches=100, patch_size=512):
    """Benchmark original inference method"""
    from UNet import Unet

    # Setup
    model = Unet(3, 3).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    transforms_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create test patches
    test_patches = [create_dummy_patch(patch_size) for _ in range(num_patches)]

    # Warmup
    _ = original_inference_single(model, test_patches[0], device, transforms_fn)

    # Benchmark
    times = []
    for patch in tqdm(test_patches, desc="Original (single)", colour="red"):
        start = time.perf_counter()
        _ = original_inference_single(model, patch, device, transforms_fn)
        times.append(time.perf_counter() - start)

    return np.array(times)

def benchmark_optimized(model_path, device, num_patches=100, patch_size=512, batch_size=12):
    """Benchmark optimized inference method"""
    from inference_optimized import OptimizedInference

    # Setup
    engine = OptimizedInference(
        model_path=model_path,
        device=str(device),
        use_fp16=torch.cuda.is_available(),
        use_jit=True,
        batch_size=batch_size
    )

    # Create test patches
    test_patches = [create_dummy_patch(patch_size) for _ in range(num_patches)]

    # Warmup
    _ = engine.predict_batch(test_patches[:batch_size])

    # Benchmark
    times = []
    for i in tqdm(range(0, len(test_patches), batch_size), desc="Optimized (batch)", colour="green"):
        batch = test_patches[i:i+batch_size]
        start = time.perf_counter()
        _ = engine.predict_batch(batch)
        elapsed = time.perf_counter() - start
        # Distribute batch time equally across patches
        times.extend([elapsed / len(batch)] * len(batch))

    return np.array(times[:num_patches])

def visualize_comparison(original_times, optimized_times, output_file='inference_comparison.png'):
    """Create visualization comparing inference speeds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Inference Optimization Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Distribution
    ax = axes[0, 0]
    ax.hist(original_times*1000, bins=30, alpha=0.6, label='Original', color='red')
    ax.hist(optimized_times*1000, bins=30, alpha=0.6, label='Optimized', color='green')
    ax.set_xlabel('Time per patch (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Speed Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Time series
    ax = axes[0, 1]
    ax.plot(original_times*1000, label='Original', marker='o', markersize=2, color='red', alpha=0.6)
    ax.plot(optimized_times*1000, label='Optimized', marker='s', markersize=2, color='green', alpha=0.6)
    ax.set_xlabel('Patch index')
    ax.set_ylabel('Time per patch (ms)')
    ax.set_title('Inference Speed Timeline')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Statistics
    ax = axes[1, 0]
    ax.axis('off')

    stats_text = f"""
    ORIGINAL METHOD
    ━━━━━━━━━━━━━━━━━━━━━━
    Mean:        {np.mean(original_times)*1000:.2f} ms
    Median:      {np.median(original_times)*1000:.2f} ms
    Std Dev:     {np.std(original_times)*1000:.2f} ms
    Min:         {np.min(original_times)*1000:.2f} ms
    Max:         {np.max(original_times)*1000:.2f} ms

    OPTIMIZED METHOD
    ━━━━━━━━━━━━━━━━━━━━━━
    Mean:        {np.mean(optimized_times)*1000:.2f} ms
    Median:      {np.median(optimized_times)*1000:.2f} ms
    Std Dev:     {np.std(optimized_times)*1000:.2f} ms
    Min:         {np.min(optimized_times)*1000:.2f} ms
    Max:         {np.max(optimized_times)*1000:.2f} ms

    IMPROVEMENT
    ━━━━━━━━━━━━━━━━━━━━━━
    Speedup:     {np.mean(original_times)/np.mean(optimized_times):.2f}x
    Time Saved:  {(np.mean(original_times)-np.mean(optimized_times))*1000:.2f} ms
    """

    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Speedup visualization
    ax = axes[1, 1]

    # Cumulative time comparison
    original_cumsum = np.cumsum(original_times)
    optimized_cumsum = np.cumsum(optimized_times)

    ax.fill_between(range(len(original_times)), original_cumsum*1000, alpha=0.3, color='red', label='Original')
    ax.fill_between(range(len(optimized_times)), optimized_cumsum*1000, alpha=0.3, color='green', label='Optimized')
    ax.plot(original_cumsum*1000, color='red', linewidth=2, label='Original (cumulative)')
    ax.plot(optimized_cumsum*1000, color='green', linewidth=2, label='Optimized (cumulative)')

    ax.set_xlabel('Number of patches processed')
    ax.set_ylabel('Cumulative time (ms)')
    ax.set_title('Cumulative Processing Time')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison chart saved to {output_file}")
    plt.close()

def print_summary(original_times, optimized_times, num_patches):
    """Print comparison summary"""
    print("\n" + "="*70)
    print("INFERENCE OPTIMIZATION BENCHMARK RESULTS".center(70))
    print("="*70)

    original_mean = np.mean(original_times)
    optimized_mean = np.mean(optimized_times)
    speedup = original_mean / optimized_mean

    print("\n📊 PER-PATCH STATISTICS")
    print("-" * 70)
    print(f"  Original Method:")
    print(f"    • Mean:        {original_mean*1000:>8.2f} ms")
    print(f"    • Median:      {np.median(original_times)*1000:>8.2f} ms")
    print(f"    • Std Dev:     {np.std(original_times)*1000:>8.2f} ms")
    print(f"    • Min/Max:     {np.min(original_times)*1000:>8.2f} / {np.max(original_times)*1000:.2f} ms")

    print(f"\n  Optimized Method:")
    print(f"    • Mean:        {optimized_mean*1000:>8.2f} ms")
    print(f"    • Median:      {np.median(optimized_times)*1000:>8.2f} ms")
    print(f"    • Std Dev:     {np.std(optimized_times)*1000:>8.2f} ms")
    print(f"    • Min/Max:     {np.min(optimized_times)*1000:>8.2f} / {np.max(optimized_times)*1000:.2f} ms")

    print(f"\n⚡ PERFORMANCE IMPROVEMENT")
    print("-" * 70)
    print(f"  Speedup:               {speedup:>8.2f}x faster")
    print(f"  Time saved per patch:  {(original_mean - optimized_mean)*1000:>8.2f} ms")
    print(f"  Time saved per 1000:   {(original_mean - optimized_mean)*1000000:>8.0f} ms")

    # Extrapolate to larger datasets
    total_patches_1k = 1000
    total_patches_57k = 57000  # Typical WSI size

    original_time_1k = original_mean * total_patches_1k
    optimized_time_1k = optimized_mean * total_patches_1k

    original_time_57k = original_mean * total_patches_57k
    optimized_time_57k = optimized_mean * total_patches_57k

    print(f"\n📈 EXTRAPOLATED PROCESSING TIME")
    print("-" * 70)
    print(f"  For 1,000 patches:")
    print(f"    • Original:  {original_time_1k:.2f}s ({original_time_1k/60:.1f} min)")
    print(f"    • Optimized: {optimized_time_1k:.2f}s ({optimized_time_1k/60:.1f} min)")
    print(f"    • Saved:     {original_time_1k - optimized_time_1k:.2f}s ({(original_time_1k - optimized_time_1k)/60:.1f} min)")

    print(f"\n  For 57,000 patches (typical WSI):")
    print(f"    • Original:  {original_time_57k:.2f}s ({original_time_57k/3600:.1f} hours)")
    print(f"    • Optimized: {optimized_time_57k:.2f}s ({optimized_time_57k/3600:.1f} hours)")
    print(f"    • Saved:     {original_time_57k - optimized_time_57k:.2f}s ({(original_time_57k - optimized_time_57k)/3600:.1f} hours)")

    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    print("\n🚀 Starting inference optimization benchmark...\n")

    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'UNet_17.pth'  # Or your model path
    num_patches = 100
    patch_size = 512
    batch_size = 12

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Test patches: {num_patches}")
    print(f"Patch size: {patch_size}×{patch_size}")
    print(f"Batch size: {batch_size}")

    try:
        # Benchmark original method
        print("\n" + "="*70)
        print("1. Benchmarking ORIGINAL method...")
        print("="*70)
        original_times = benchmark_original(model_path, device, num_patches, patch_size)

        # Benchmark optimized method
        print("\n" + "="*70)
        print("2. Benchmarking OPTIMIZED method...")
        print("="*70)
        optimized_times = benchmark_optimized(model_path, device, num_patches, patch_size, batch_size)

        # Print summary
        print_summary(original_times, optimized_times, num_patches)

        # Create visualization
        print("Generating comparison chart...")
        visualize_comparison(original_times, optimized_times)

    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print(f"   Please update the model_path in the script.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
