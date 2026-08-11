#!/usr/bin/env python3
"""
Checker script to test if TorchTensorParallelPlugin can be imported from accelerate.
Run this in the test venv after installing accelerate 0.33.0+
"""

import sys

print("=" * 70)
print("ACCELERATE IMPORT CHECKER")
print("=" * 70)

# Check accelerate version
try:
    import accelerate
    print(f"✓ accelerate installed: {accelerate.__version__}")
except ImportError as e:
    print(f"✗ accelerate not installed: {e}")
    sys.exit(1)

# Check if TorchTensorParallelPlugin exists in standard location
plugin_found = False
try:
    from accelerate.utils import TorchTensorParallelPlugin
    print(f"✓ TorchTensorParallelPlugin found in accelerate.utils")
    plugin_found = True
except ImportError as e:
    print(f"✗ TorchTensorParallelPlugin NOT in accelerate.utils")
    print(f"  Error: {e}")

    # Try alternative import locations
    try:
        from accelerate import TorchTensorParallelPlugin
        print(f"✓ TorchTensorParallelPlugin found in accelerate (root)")
        plugin_found = True
    except ImportError:
        print(f"✗ Not in accelerate root either")

if not plugin_found:
    print("\n" + "=" * 70)
    print("VERSION MISMATCH DETECTED")
    print("=" * 70)
    print("\nPossible solutions:")
    print("1. Update accelerate to latest version:")
    print("   pip install --upgrade accelerate")
    print("\n2. OR downgrade transformers to compatible version:")
    print("   pip install 'transformers<4.50.0'")
    print("\n3. Check version compatibility:")
    print("   pip show accelerate transformers")
    print("\n" + "=" * 70)
    sys.exit(1)

# Try importing transformers to see if it works now
try:
    from transformers import Trainer
    print(f"✓ transformers.Trainer can be imported")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")
    print("\nThis is likely the root cause of your metric computation error.")
    sys.exit(1)

# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ sentence_transformers.SentenceTransformer can be imported")
except ImportError as e:
    print(f"✗ sentence_transformers import failed: {e}")
    sys.exit(1)

print("=" * 70)
print("✓ ALL IMPORTS SUCCESSFUL - Your environment is compatible!")
print("=" * 70)
print("\nYou can now run your training/evaluation code without import errors.")
