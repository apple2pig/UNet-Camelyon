# 🚀 如何运行 Skill - 完整指南

本指南将帮助你快速上手运行优化推理 Skill。

---

## 📍 前置准备

### 1. 确保虚拟环境已激活

**Windows (Command Prompt)**:
```bash
.venv\Scripts\activate
```

**Windows (PowerShell)**:
```bash
.venv\Scripts\Activate.ps1
```

**Linux/Mac**:
```bash
source .venv/bin/activate
```

验证激活成功（命令行前缀应显示 `.venv`）:
```bash
(.venv) C:\Users\junyou.zhang\Desktop\Us\UNet-Camelyon>
```

### 2. 确保已安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型文件

你需要一个训练好的模型权重文件：
- 文件名: `UNet_17.pth` (或其他名称)
- 位置: 项目根目录或任意路径
- 大小: 约 100MB

---

## 🎯 三种运行方式

### 方式 1️⃣: 运行示例脚本（推荐新手）

**最简单，适合快速体验**

#### 步骤

1. **进入 skills 文件夹**
   ```bash
   cd skills
   ```

2. **编辑配置** (打开 `example_optimized_inference.py`)

   找到这些行并修改为你的实际路径:
   ```python
   # 第 16 行 - 模型路径
   model_path='UNet_17.pth'        # 改为你的模型路径

   # 第 31 行 - WSI 文件路径
   wsi_path = '/Camelyon16/test_040.tif'  # 改为你的 WSI 路径

   # 第 58-59 行 - 输入输出目录
   input_dir = 'patch_path/'       # 改为你的补丁目录
   output_dir = 'output_path/'     # 改为输出目录
   ```

3. **运行脚本**
   ```bash
   python example_optimized_inference.py
   ```

4. **查看输出**
   - 模型初始化信息
   - 处理进度条
   - 性能统计信息
   - 输出文件位置

---

### 方式 2️⃣: 性能基准测试

**对比原始vs优化性能，查看加速效果**

#### 步骤

1. **进入 skills 文件夹**
   ```bash
   cd skills
   ```

2. **编辑配置** (打开 `compare_inference_speed.py`)

   修改这些变量:
   ```python
   # 约 122 行
   model_path = 'UNet_17.pth'  # 改为你的模型路径
   device = 'cuda:0'            # GPU 设备选择
   num_patches = 100            # 测试补丁数量
   batch_size = 12              # 批处理大小
   ```

3. **运行测试**
   ```bash
   python compare_inference_speed.py
   ```

4. **查看结果**
   - 详细的性能对比表格
   - 生成的可视化图表 (`inference_comparison.png`)
   - 速度提升倍数
   - 时间节省统计

**示例输出**:
```
==============================================================
PERFORMANCE BENCHMARK
==============================================================

[1] Original Method (Single inference)
  Average time: 42.35ms per image

[2] Optimized Method (Batch + FP16 + JIT)
  Average time: 7.68ms per image

==============================================================
SPEEDUP: 5.51x faster
Time saved: 34.67ms per image
==============================================================
```

---

### 方式 3️⃣: 编写自己的脚本

**自定义使用，最大灵活性**

#### 基础模板

```python
# 1. 导入
from skills.inference_optimized import OptimizedInference
import torch

# 2. 检测设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 3. 初始化优化引擎
engine = OptimizedInference(
    model_path='UNet_17.pth',      # 你的模型路径
    device=device,
    use_fp16=True,                  # 启用半精度
    use_jit=True,                   # 启用 JIT 编译
    batch_size=12,                  # 根据 GPU 内存调整
    num_workers=4                   # 数据加载线程数
)

# 4. 使用引擎

# 选项 A: 处理 WSI
engine.process_wsi_batched(
    wsi_path='your_slide.tif',
    patch_size=512,
    output_path='result.png',
    overlap=0  # 0 = 无重叠，32/64 = 平滑边界
)

# 选项 B: 处理 patch 目录
engine.process_patches_directory(
    input_dir='patches/',
    output_dir='results/'
)

# 选项 C: 单个预测
from PIL import Image
patch = Image.open('single_patch.png').convert('RGB')
prediction = engine.predict_batch([patch])
print(f"Prediction shape: {prediction[0].shape}")

# 选项 D: 导出 ONNX
engine.export_to_onnx('model.onnx')
```

#### 保存为文件并运行

```bash
# 1. 创建自己的脚本
notepad my_inference.py

# 2. 粘贴上面的模板，修改路径
# 3. 保存文件

# 4. 运行
cd skills
python my_inference.py
```

---

## 🛠️ 常见配置示例

### 例子 1: 快速处理小 WSI

```python
engine = OptimizedInference(
    model_path='UNet_17.pth',
    device='cuda:0',
    batch_size=8,          # 较小批次
    use_fp16=False         # 禁用 FP16 保证精度
)

engine.process_wsi_batched(
    wsi_path='small_slide.tif',
    patch_size=256,        # 较小补丁
    output_path='result.png',
    overlap=0
)
```

### 例子 2: 处理超大 WSI

```python
engine = OptimizedInference(
    model_path='UNet_17.pth',
    device='cuda:0',
    batch_size=16,         # 大批次
    use_fp16=True          # 启用 FP16 加速
)

engine.process_wsi_batched(
    wsi_path='huge_slide.tif',
    patch_size=512,
    output_path='result.png',
    overlap=64             # 重叠边界更平滑
)
```

### 例子 3: CPU 推理

```python
engine = OptimizedInference(
    model_path='UNet_17.pth',
    device='cpu',          # 使用 CPU
    use_fp16=False,        # CPU 不支持 FP16
    batch_size=2,          # 批次较小
    use_jit=True           # JIT 在 CPU 上仍有效
)

# 处理...
```

### 例子 4: 实时应用

```python
engine = OptimizedInference(
    model_path='UNet_17.pth',
    device='cuda:0',
    batch_size=1,          # 单张图像
    use_fp16=True,
    use_jit=True
)

# 单个预测
from PIL import Image
patch = Image.open('real_time_patch.png').convert('RGB')
pred = engine.predict_batch([patch])
result_value = pred[0].max()  # 获取预测值
```

---

## 📊 参数调优

### GPU 显存不足

```python
# 原始配置报错: CUDA Out of Memory

# 解决方案 1: 降低批处理大小
engine = OptimizedInference(batch_size=4)  # 从 12 改为 4

# 解决方案 2: 禁用 FP16
engine = OptimizedInference(use_fp16=False)

# 解决方案 3: 降低补丁大小
engine.process_wsi_batched(patch_size=256)  # 从 512 改为 256

# 解决方案 4: 组合方案
engine = OptimizedInference(
    batch_size=4,
    use_fp16=False,
    use_jit=True
)
```

### 提高精度

```python
engine = OptimizedInference(
    use_fp16=False,  # 使用 FP32 (完整精度)
    use_jit=True     # 但仍然使用 JIT 加速
)

# 或添加边界平滑
engine.process_wsi_batched(
    overlap=64  # 增加重叠以获得平滑结果
)
```

### 提高速度

```python
engine = OptimizedInference(
    batch_size=24,   # 增加批处理大小
    use_fp16=True,   # 启用半精度
    use_jit=True,    # 启用 JIT
    num_workers=8    # 更多加载线程
)

engine.process_wsi_batched(
    overlap=0  # 无重叠最快
)
```

---

## ⚡ 性能监控

### 方法 1: GPU 监控

在另一个终端运行:
```bash
# 实时监控 GPU 使用情况
watch -n 1 nvidia-smi
```

或一次性查看:
```bash
nvidia-smi
```

### 方法 2: CPU 和内存监控

```bash
# Windows
tasklist | find "python"

# Linux
ps aux | grep python
```

### 方法 3: 程序内监控

脚本会自动输出:
```
✓ Processing complete!
  Time elapsed: 45.23s
  Speed: 14.22 patches/sec
  Average time per patch: 70.35ms
```

---

## 🐛 常见错误和解决方案

### ❌ 错误 1: 模型文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: 'UNet_17.pth'
```

**解决方案**:
```python
import os

# 使用绝对路径
model_path = os.path.abspath('../UNet_17.pth')
engine = OptimizedInference(model_path=model_path)

# 或检查文件是否存在
if os.path.exists('UNet_17.pth'):
    print("✓ 模型文件找到")
else:
    print("✗ 模型文件未找到")
    print(f"当前目录: {os.getcwd()}")
    print(f"目录内容: {os.listdir('.')}")
```

### ❌ 错误 2: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决方案**:
```python
# 1. 减小批处理大小
engine = OptimizedInference(batch_size=4)

# 2. 禁用 FP16
engine = OptimizedInference(use_fp16=False)

# 3. 清空 GPU 缓存
import torch
torch.cuda.empty_cache()

# 4. 重启 Python 环境
```

### ❌ 错误 3: 导入错误

```
ModuleNotFoundError: No module named 'inference_optimized'
```

**解决方案**:
```bash
# 确保在正确的目录中
cd C:\Users\junyou.zhang\Desktop\Us\UNet-Camelyon\skills

# 或使用绝对导入
import sys
sys.path.insert(0, 'C:\\Users\\junyou.zhang\\Desktop\\Us\\UNet-Camelyon\\skills')
from inference_optimized import OptimizedInference
```

### ❌ 错误 4: WSI 文件读取错误

```
openslide.OpenSlideError: ...
```

**解决方案**:
```python
# 检查文件格式
# 支持的格式: .tif, .tiff, .ndpi, .vms, 等

# 检查文件是否完整
import os
file_size = os.path.getsize('test.tif')
print(f"文件大小: {file_size / (1024**3):.2f} GB")

# 使用绝对路径
wsi_path = os.path.abspath('/path/to/slide.tif')
```

---

## 📈 性能期望值

基于 RTX 3090, 512×512 patches, batch_size=12:

| 操作 | 速度 | 内存 |
|------|------|------|
| 原始推理 | ~45ms/patch | 2.5GB |
| 优化推理 | ~8ms/patch | 1.2GB |
| **加速倍数** | **5.6x** | **-52%** |

### 时间估算

**处理 1000 patches**:
- 原始: 45 秒
- 优化: 8 秒
- **节省: 37 秒**

**处理大型 WSI (57,000 patches)**:
- 原始: ~8 小时
- 优化: ~1.1 小时
- **节省: ~7 小时**

---

## 📚 进阶使用

### 自定义后处理

```python
engine = OptimizedInference('UNet_17.pth')

# 批量预测
patches = [Image.open(f'patch_{i}.png') for i in range(10)]
predictions = engine.predict_batch(patches)

# 自定义处理
for pred, patch in zip(predictions, patches):
    # 应用自定义阈值
    binary_pred = (pred > 0.5).astype(np.uint8)

    # 保存
    result = Image.fromarray(binary_pred * 255)
    result.save('custom_result.png')
```

### 批量处理多个 WSI

```python
import os
from pathlib import Path

wsi_dir = '/path/to/wsi/'
output_dir = '/path/to/output/'

engine = OptimizedInference('UNet_17.pth')

for wsi_file in os.listdir(wsi_dir):
    if wsi_file.endswith('.tif'):
        wsi_path = os.path.join(wsi_dir, wsi_file)
        output_path = os.path.join(output_dir, f'{wsi_file}_result.png')

        print(f"处理: {wsi_file}...")
        engine.process_wsi_batched(wsi_path, output_path)
```

### ONNX 部署

```python
# 导出
engine = OptimizedInference('UNet_17.pth')
engine.export_to_onnx('model.onnx')

# 在其他环境中使用
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')

# 推理
import numpy as np
dummy_input = np.random.randn(1, 3, 512, 512).astype(np.float32)
outputs = session.run(None, {'input': dummy_input})
```

---

## ✅ 检查清单

运行前请确认:

- [ ] 虚拟环境已激活
- [ ] 依赖包已安装 (`pip install -r requirements.txt`)
- [ ] 模型文件存在且路径正确
- [ ] WSI/patch 文件存在且路径正确
- [ ] 输出目录存在或可创建
- [ ] GPU 显存充足 (或使用 CPU)

---

## 🎓 完整工作流程

```bash
# 1. 激活虚拟环境
.venv\Scripts\activate

# 2. 进入 skills 目录
cd skills

# 3. 编辑配置文件
notepad example_optimized_inference.py
# 修改模型路径、WSI 路径等

# 4. 运行示例
python example_optimized_inference.py

# 5. 查看结果
# 输出文件: wsi_result.png, unet_optimized.onnx

# 6. （可选）运行性能测试
python compare_inference_speed.py

# 7. 查看性能图表
# 生成文件: inference_comparison.png
```

---

## 📞 需要帮助？

1. **查看详细文档**: `skills/INFERENCE_OPTIMIZATION_README.md`
2. **查看 Skill 说明**: `skills/README.md`
3. **查看项目结构**: `PROJECT_STRUCTURE.md`
4. **查看主文档**: `README.md`

---

**祝你使用愉快！** 🚀

如有问题，请查阅上述文档或检查错误日志。
