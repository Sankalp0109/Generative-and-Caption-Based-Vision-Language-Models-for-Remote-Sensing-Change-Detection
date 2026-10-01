#!/usr/bin/env python3
"""
Verify PyTorch + dependencies are correctly installed for GTX 1080 Ti (sm_61).
Run this after: pip install -r requirements.txt
"""

import sys
import warnings

warnings.filterwarnings('ignore')

print("\n" + "=" * 70)
print("PHASE 8 SETUP VERIFICATION")
print("=" * 70)

# Check 1: PyTorch Version
print("\n✓ Checking PyTorch...")
try:
    import torch
    version = torch.__version__
    major, minor = int(version.split('.')[0]), int(version.split('.')[1])

    if major == 2 and minor == 0:
        print(f"  ✅ PyTorch {version} (CORRECT for sm_61)")
    elif major == 2 and minor >= 1:
        print(f"  ❌ PyTorch {version} (TOO NEW - dropped sm_61 support!)")
        print(f"     Fix: pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    else:
        print(f"  ⚠️  PyTorch {version} (unexpected version)")
except ImportError:
    print("  ❌ PyTorch not installed")
    sys.exit(1)

# Check 2: CUDA Availability
print("\n✓ Checking CUDA...")
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"  ✅ GPU detected: {device}")
    print(f"  ✅ Compute capability: {capability[0]}.{capability[1]} (sm_{capability[0]}{capability[1]})")

    if capability == (6, 1):
        print(f"  ✅ GTX 1080 Ti confirmed - PyTorch 2.0.1 is correct!")
    elif capability[0] >= 7:
        print(f"  ℹ️  Newer GPU detected (sm_{capability[0]}{capability[1]})")
        print(f"     Could use PyTorch 2.1+ but 2.0.1 is fine")
else:
    print(f"  ⚠️  CUDA not available - will run on CPU")
    print(f"     If GPU available, check: pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121")

# Check 3: Critical Dependencies
print("\n✓ Checking critical dependencies...")
packages_to_check = [
    ("transformers", "4.40"),
    ("sentence_transformers", "3.0"),
    ("open_clip_torch", "0.24"),
    ("bitsandbytes", "0.41"),
    ("accelerate", "0.27"),
]

all_ok = True
for package, expected_version in packages_to_check:
    try:
        mod = __import__(package.replace('-', '_'))
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✅ {package:<25} {version}")
    except ImportError:
        print(f"  ❌ {package:<25} NOT INSTALLED")
        all_ok = False

# Check 4: Summary
print("\n" + "=" * 70)
if all_ok and torch.__version__.startswith("2.0"):
    print("✅ SETUP VERIFIED - Ready for training!")
    print("\nNext steps:")
    print("  1. Run: python check_vram_calibration.py")
    print("  2. Run: jupyter notebook phase8_finetune.ipynb")
else:
    print("❌ SETUP INCOMPLETE - Fix issues above")
    sys.exit(1)

print("=" * 70 + "\n")
