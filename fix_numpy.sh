#!/usr/bin/env bash
# Quick fix for NumPy 2.x → 1.x compatibility issue

echo "Fixing NumPy compatibility for PyTorch 2.0.1..."
echo ""

# Uninstall NumPy 2.x
echo "Uninstalling NumPy 2.x..."
pip uninstall numpy -y

# Install NumPy 1.x
echo "Installing NumPy 1.x (compatible with PyTorch 2.0.1)..."
pip install 'numpy<2'

# Verify
echo ""
echo "Verifying installation..."
python << 'PYEOF'
import numpy
import torch

print(f"✅ NumPy: {numpy.__version__}")
if numpy.__version__.startswith('1'):
    print("✅ NumPy version OK (1.x)")
else:
    print(f"❌ ERROR: NumPy {numpy.__version__} is not 1.x!")
    import sys
    sys.exit(1)

print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print("✅ Ready for training!")
else:
    print("⚠️  GPU not available")
PYEOF

echo ""
echo "✅ NumPy fix complete!"
