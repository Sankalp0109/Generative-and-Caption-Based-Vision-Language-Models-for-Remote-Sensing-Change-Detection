#!/usr/bin/env python3
"""
Comprehensive compatibility check for Phase 8 training.
Verifies all critical dependencies and GPU setup.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("\n" + "=" * 80)
print("PHASE 8 COMPATIBILITY CHECK".center(80))
print("=" * 80 + "\n")

# Track results
checks_passed = 0
checks_failed = 0

def check(name, condition, expected=None, actual=None):
    global checks_passed, checks_failed
    if condition:
        print(f"✅ {name}")
        if actual:
            print(f"   → {actual}")
        checks_passed += 1
    else:
        print(f"❌ {name}")
        if expected and actual:
            print(f"   Expected: {expected}")
            print(f"   Got: {actual}")
        checks_failed += 1

# ========== 1. PYTORCH CHECK ==========
print("1. PyTorch Version Check")
print("-" * 80)
try:
    import torch
    pytorch_version = torch.__version__
    is_2_0 = pytorch_version.startswith('2.0')
    check("PyTorch installed", True, actual=f"Version: {pytorch_version}")
    check("PyTorch 2.0.1 (required for sm_61)", is_2_0, "2.0.x", pytorch_version)
    
    if not is_2_0:
        print(f"\n⚠️  WARNING: PyTorch {pytorch_version} may not support GTX 1080 Ti (sm_61)")
        print("   Solution: Reinstall PyTorch 2.0.1")
except ImportError as e:
    check("PyTorch installed", False)
    print(f"   Error: {e}")

# ========== 2. NUMPY CHECK ==========
print("\n2. NumPy Version Check")
print("-" * 80)
try:
    import numpy
    numpy_version = numpy.__version__
    numpy_major = int(numpy_version.split('.')[0])
    is_numpy_1x = numpy_major == 1
    check("NumPy installed", True, actual=f"Version: {numpy_version}")
    check("NumPy 1.x (PyTorch 2.0.1 requires 1.x)", is_numpy_1x, "1.x", f"{numpy_major}.x")
    
    if not is_numpy_1x:
        print(f"\n⚠️  WARNING: NumPy {numpy_version} is incompatible with PyTorch 2.0.1")
        print("   Solution: pip install 'numpy<2'")
except ImportError as e:
    check("NumPy installed", False)

# ========== 3. CORE DEPENDENCIES CHECK ==========
print("\n3. Core Dependency Imports")
print("-" * 80)

packages = [
    ('transformers', 'Transformers'),
    ('sentence_transformers', 'Sentence-Transformers'),
    ('open_clip', 'OpenCLIP'),
    ('bitsandbytes', 'BitsAndBytes'),
    ('accelerate', 'Accelerate'),
]

for module_name, display_name in packages:
    try:
        mod = __import__(module_name.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        check(f"{display_name} installed", True, actual=f"Version: {version}")
    except ImportError as e:
        check(f"{display_name} installed", False)
        print(f"   Error: {e}")

# ========== 4. JUPYTER CHECK ==========
print("\n4. Jupyter Installation")
print("-" * 80)

jupyter_modules = [
    ('jupyter', 'Jupyter'),
    ('nbconvert', 'NBConvert'),
    ('ipykernel', 'IPyKernel'),
]

for module_name, display_name in jupyter_modules:
    try:
        mod = __import__(module_name.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        check(f"{display_name} installed", True, actual=f"Version: {version}")
    except ImportError as e:
        check(f"{display_name} installed", False)

# ========== 5. CUDA CHECK ==========
print("\n5. CUDA & GPU Check")
print("-" * 80)

try:
    cuda_available = torch.cuda.is_available()
    check("CUDA available", cuda_available)
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        check("GPU detected", True, actual=gpu_name)
        check("Compute capability reported", True, actual=f"sm_{gpu_capability[0]}{gpu_capability[1]}")
        check("GTX 1080 Ti detected", "GTX 1080 Ti" in gpu_name or "1080" in gpu_name.lower())
        check(f"GPU memory", True, actual=f"{gpu_memory:.2f} GB")
        
        # Test CUDA operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            check("CUDA operations work", True, actual="Matrix multiplication successful")
        except RuntimeError as e:
            check("CUDA operations work", False)
            print(f"   Error: {e}")
    else:
        print("\n⚠️  WARNING: CUDA not available - will run on CPU (100x slower)")
        print("   Solution: Reinstall PyTorch with CUDA support")
        print("   Command: pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118")
        
except Exception as e:
    check("CUDA check", False)
    print(f"   Error: {e}")

# ========== 6. VERSION COMPATIBILITY MATRIX ==========
print("\n6. Compatibility Matrix")
print("-" * 80)

try:
    import transformers
    import sentence_transformers
    
    transf_version = transformers.__version__
    sent_version = sentence_transformers.__version__
    
    transf_major = int(transf_version.split('.')[0])
    sent_major = int(sent_version.split('.')[0])
    
    # Check ranges
    transf_ok = 4 <= transf_major < 5
    sent_ok = 2 <= sent_major < 3
    
    check("Transformers in compatible range (4.x)", transf_ok, "4.x", f"{transf_major}.x")
    check("Sentence-Transformers in compatible range (2.x)", sent_ok, "2.x", f"{sent_major}.x")
    
except Exception as e:
    print(f"⚠️  Could not check compatibility matrix: {e}")

# ========== SUMMARY ==========
print("\n" + "=" * 80)
print("SUMMARY".center(80))
print("=" * 80)
print(f"Checks passed: {checks_passed}")
print(f"Checks failed: {checks_failed}")

if checks_failed == 0:
    print("\n✅ ALL CHECKS PASSED - Ready for training!")
    sys.exit(0)
else:
    print(f"\n❌ {checks_failed} checks failed - Fix issues before training")
    sys.exit(1)

