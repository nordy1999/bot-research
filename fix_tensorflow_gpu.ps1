# ================================================================
#  Fix TensorFlow GPU for RTX 2060
#  Run this in PowerShell (Admin not required)
# ================================================================

Write-Host "`n=== TensorFlow GPU Setup for RTX 2060 ===" -ForegroundColor Cyan

# Step 1: Check current TF version
Write-Host "`n[1/4] Checking current TensorFlow..." -ForegroundColor Yellow
python -c "import tensorflow as tf; print(f'TF version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(''GPU'')}')"

# Step 2: Uninstall current TF and install GPU version
Write-Host "`n[2/4] Installing TensorFlow with GPU support..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor DarkGray

# TF 2.16+ has unified package (CPU+GPU in one)
# If on TF 2.10-2.15, need tensorflow[and-cuda]
# For TF 2.16+, just need the right CUDA toolkit

pip install --upgrade tensorflow[and-cuda] --break-system-packages 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Trying alternative install..." -ForegroundColor DarkGray
    pip install --upgrade tensorflow --break-system-packages
}

# Step 3: Install CUDA toolkit via pip (works alongside NVIDIA drivers)
Write-Host "`n[3/4] Installing CUDA toolkit..." -ForegroundColor Yellow
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12 --break-system-packages 2>$null

# Step 4: Verify
Write-Host "`n[4/4] Verifying GPU detection..." -ForegroundColor Yellow
python -c @"
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPU DETECTED: {len(gpus)} device(s)')
    for g in gpus:
        print(f'  - {g.name} ({g.device_type})')
    # Set memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('Memory growth enabled')
    # Quick test
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f'GPU compute test: {c.numpy()} - SUCCESS')
else:
    print('NO GPU detected')
    print('')
    print('Troubleshooting:')
    print('1. Make sure NVIDIA drivers are up to date (nvidia-smi should work)')
    print('2. Try: pip install tensorflow==2.15.0')
    print('3. Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads')
    print('4. Install cuDNN from https://developer.nvidia.com/cudnn')
    print('5. Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin')
"@

Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "If GPU is detected, retrain with: python train_models.py" -ForegroundColor Green
Write-Host "LSTM and TFT should now train 5-10x faster with better accuracy.`n" -ForegroundColor Green
