#!/usr/bin/env bash
export HOME=/home/user
export PYTHONUNBUFFERED=1
export HF_HOME=/home/user/.cache/huggingface

export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)

export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Disable audio warnings in Docker
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

# ═══════════════════════════ CUDA DEBUG CHECKS ═══════════════════════════

echo "🔍 CUDA Environment Debug Information:"
echo "═══════════════════════════════════════════════════════════════════════"

# Check CUDA driver on host (if accessible)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ nvidia-smi available"
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "❌ nvidia-smi failed to query GPU"
    echo "🏃 Running Processes:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "ℹ️  No running CUDA processes"
else
    echo "❌ nvidia-smi not available in container"
fi

# Check CUDA runtime libraries
echo ""
echo "🔧 CUDA Runtime Check:"
if ls /usr/local/cuda*/lib*/libcudart.so* >/dev/null 2>&1; then
    echo "✅ CUDA runtime libraries found:"
    ls /usr/local/cuda*/lib*/libcudart.so* 2>/dev/null
else
    echo "❌ CUDA runtime libraries not found"
fi

# Check CUDA devices
echo ""
echo "🖥️  CUDA Device Files:"
if ls /dev/nvidia* >/dev/null 2>&1; then
    echo "✅ NVIDIA device files found:"
    ls -la /dev/nvidia* 2>/dev/null
else
    echo "❌ No NVIDIA device files found - Docker may not have GPU access"
fi

# Check CUDA environment variables
echo ""
echo "🌍 CUDA Environment Variables:"
echo "   CUDA_HOME: ${CUDA_HOME:-not set}"
echo "   CUDA_ROOT: ${CUDA_ROOT:-not set}"
echo "   CUDA_PATH: ${CUDA_PATH:-not set}"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "   TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-not set}"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Check PyTorch CUDA availability
echo ""
echo "🐍 PyTorch CUDA Check:"
python3 -c "
import sys
try:
    import torch
    print('✅ PyTorch imported successfully')
    print(f'   Version: {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA version: {torch.version.cuda}')
        print(f'   cuDNN version: {torch.backends.cudnn.version()}')
        print(f'   Device count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'   Device {i}: {props.name} (SM {props.major}.{props.minor}, {props.total_memory//1024//1024}MB)')
    else:
        print('❌ CUDA not available to PyTorch')
        print('   This could mean:')
        print('   - CUDA runtime not properly installed')
        print('   - GPU not accessible to container')
        print('   - Driver/runtime version mismatch')
except ImportError as e:
    print(f'❌ Failed to import PyTorch: {e}')
except Exception as e:
    print(f'❌ PyTorch CUDA check failed: {e}')
" 2>&1

# Check for common CUDA issues
echo ""
echo "🩺 Common Issue Diagnostics:"

# Check if running with proper Docker flags
if [ ! -e /dev/nvidia0 ] && [ ! -e /dev/nvidiactl ]; then
    echo "❌ No NVIDIA device nodes - container likely missing --gpus all or --runtime=nvidia"
fi

# Check CUDA library paths
if [ -z "$LD_LIBRARY_PATH" ] || ! echo "$LD_LIBRARY_PATH" | grep -q cuda; then
    echo "⚠️  LD_LIBRARY_PATH may not include CUDA libraries"
fi

# Check permissions on device files
if ls /dev/nvidia* >/dev/null 2>&1; then
    if ! ls -la /dev/nvidia* | grep -q "rw-rw-rw-\|rw-r--r--"; then
        echo "⚠️  NVIDIA device files may have restrictive permissions"
    fi
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 Starting application..."
echo ""
exec python3 /workspace/wangp_server.py
echo "Finished "
