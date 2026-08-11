#!/usr/bin/env python3
"""
Comprehensive bitsandbytes + CUDA diagnostic script
Run this remotely to identify the root cause of bitsandbytes failures
"""

import sys
import os
import subprocess
import json

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def run_cmd(cmd):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

# ============================================================================
# 1. Python & Environment
# ============================================================================
print_section("1. PYTHON & ENVIRONMENT")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Python Prefix (venv?): {sys.prefix}")
print(f"PATH: {os.environ.get('PATH', 'NOT SET')}")

# ============================================================================
# 2. CUDA Availability
# ============================================================================
print_section("2. CUDA AVAILABILITY")

# Check nvidia-smi
stdout, stderr, code = run_cmd("nvidia-smi --version")
if code == 0:
    print(f"✓ nvidia-smi found: {stdout}")
else:
    print(f"✗ nvidia-smi NOT found: {stderr}")

# Check CUDA_HOME
cuda_home = os.environ.get("CUDA_HOME", "NOT SET")
print(f"CUDA_HOME: {cuda_home}")

# Check common CUDA locations
cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/local/cuda-11", "/usr/local/cuda-12"]
for path in cuda_paths:
    if os.path.exists(path):
        print(f"  Found CUDA at: {path}")

# ============================================================================
# 3. LD_LIBRARY_PATH
# ============================================================================
print_section("3. LD_LIBRARY_PATH")
ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "NOT SET")
print(f"LD_LIBRARY_PATH: {ld_lib_path}")

if ld_lib_path != "NOT SET":
    paths = ld_lib_path.split(":")
    print(f"Number of paths: {len(paths)}")
    for i, path in enumerate(paths[:5], 1):
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {path}")
    if len(paths) > 5:
        print(f"  ... and {len(paths)-5} more paths")

# Check for CUDA libraries in common locations
print("\nSearching for CUDA libraries...")
cuda_lib_paths = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-11/lib64",
    "/usr/local/cuda-12/lib64",
    "/opt/cuda/lib64",
]
for path in cuda_lib_paths:
    if os.path.exists(path):
        libs = [f for f in os.listdir(path) if "cublas" in f.lower() or "cuda" in f.lower()][:3]
        if libs:
            print(f"  ✓ Found in {path}: {libs}")

# ============================================================================
# 4. PyTorch Installation
# ============================================================================
print_section("4. PYTORCH INSTALLATION")
try:
    import torch
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  Current CUDA Device: {torch.cuda.current_device()}")
        print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch CUDA Compiled: {torch.version.cuda}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

# ============================================================================
# 5. bitsandbytes Installation & Diagnostics
# ============================================================================
print_section("5. BITSANDBYTES INSTALLATION")
try:
    import bitsandbytes
    print(f"✓ bitsandbytes Version: {bitsandbytes.__version__}")
except ImportError as e:
    print(f"✗ bitsandbytes import failed (full error below)")
    print(f"  Error: {e}")
    print("\n  Attempting staged import to find exact failure point...")

# Staged import test
print("\nStaged import diagnostics:")
stages = [
    ("import bitsandbytes", "import bitsandbytes"),
    ("import bitsandbytes.cuda_setup", "import bitsandbytes.cuda_setup"),
    ("import bitsandbytes.utils", "import bitsandbytes.utils"),
    ("from bitsandbytes import functional", "from bitsandbytes import functional"),
]

for stage_name, import_stmt in stages:
    try:
        exec(import_stmt)
        print(f"  ✓ {stage_name}")
    except Exception as e:
        print(f"  ✗ {stage_name}")
        print(f"     Error: {type(e).__name__}: {str(e)[:100]}")

# ============================================================================
# 6. Transformers Integration
# ============================================================================
print_section("6. TRANSFORMERS INTEGRATION")
try:
    import transformers
    print(f"✓ Transformers Version: {transformers.__version__}")

    # Try to import quantization config
    try:
        from transformers import BitsAndBytesConfig
        print(f"✓ BitsAndBytesConfig available")
    except ImportError as e:
        print(f"✗ BitsAndBytesConfig import failed: {e}")

except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

# ============================================================================
# 7. Package Compatibility Matrix
# ============================================================================
print_section("7. PACKAGE COMPATIBILITY MATRIX")
packages = {
    "torch": None,
    "torchvision": None,
    "transformers": None,
    "bitsandbytes": None,
    "accelerate": None,
    "open-clip-torch": None,
}

for pkg_name in packages:
    try:
        mod = __import__(pkg_name.replace("-", "_"))
        version = getattr(mod, "__version__", "unknown")
        packages[pkg_name] = version
        print(f"  {pkg_name:20s} = {version}")
    except ImportError:
        packages[pkg_name] = "NOT INSTALLED"
        print(f"  {pkg_name:20s} = NOT INSTALLED")

# ============================================================================
# 8. Quick Fix Suggestions
# ============================================================================
print_section("8. SUGGESTED FIXES (in order of likelihood)")

issues_found = []

if not torch.cuda.is_available():
    issues_found.append(("CUDA not available to PyTorch",
                        "Reinstall PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"))

if ld_lib_path == "NOT SET" or "/cuda" not in ld_lib_path:
    issues_found.append(("LD_LIBRARY_PATH missing CUDA libraries",
                        "Add CUDA to path: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"))

if packages["bitsandbytes"] == "NOT INSTALLED":
    issues_found.append(("bitsandbytes not installed",
                        "Install bitsandbytes: pip install bitsandbytes>=0.43.0"))
else:
    issues_found.append(("bitsandbytes CUDA backend issue",
                        "Try: pip install --upgrade --force-reinstall bitsandbytes>=0.43.1"))

if packages["transformers"] and packages["bitsandbytes"]:
    # Check version compatibility
    if packages["transformers"].startswith("4.4") and packages["bitsandbytes"].startswith("0.42"):
        issues_found.append(("Version incompatibility detected",
                            "Upgrade: pip install transformers>=4.45.0 bitsandbytes>=0.43.1"))

if issues_found:
    for i, (issue, fix) in enumerate(issues_found, 1):
        print(f"\n{i}. {issue}")
        print(f"   Fix: {fix}")
else:
    print("✓ No obvious issues detected. Environment looks compatible.")

# ============================================================================
# 9. Final Test
# ============================================================================
print_section("9. FINAL INTEGRATION TEST")
print("\nAttempting to create BitsAndBytesConfig + load quantized model...")
try:
    import torch
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("✓ BitsAndBytesConfig created successfully")
    print("  Next step: Try loading a small model (e.g., 'gpt2') with this config")
except Exception as e:
    print(f"✗ Test failed: {type(e).__name__}: {e}")

print_section("END OF DIAGNOSTICS")
print("\n📋 SUMMARY:")
print(f"  PyTorch CUDA available: {torch.cuda.is_available() if 'torch' in dir() else 'unknown'}")
print(f"  bitsandbytes installed: {packages.get('bitsandbytes', 'unknown')}")
print(f"  LD_LIBRARY_PATH set: {ld_lib_path != 'NOT SET'}")
print("\n✉️  Share this output with your advisor/support team for diagnosis.")
