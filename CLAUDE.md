# Project Setup & Dependencies

## 🔧 Critical: GTX 1080 Ti Compatibility

**Problem**: GTX 1080 Ti has compute capability **sm_61**, which PyTorch 2.1+ NO LONGER supports.

**Solution**: This project is locked to **PyTorch 2.0.1** for sm_61 compatibility. DO NOT upgrade PyTorch.

## ✅ Verified Working Configuration

| Component | Version | Status |
|-----------|---------|--------|
| PyTorch | 2.0.1 | ✅ Works with sm_61 |
| CUDA | 12.8 | ✅ Verified |
| GPU | GTX 1080 Ti | ✅ sm_61 compatible |
| Python | 3.10.12 | ✅ Tested |

## 🚀 Installation (One-time Setup)

### 1. Create Fresh Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies (EXACT versions)
```bash
# Install FROM requirements.txt WITH correct PyTorch index
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA {torch.cuda.get_device_name(0)}')"
```

### 3. Verify GPU Access
```bash
python check_vram_calibration.py
```

Expected output:
```
✓ GPU available: NVIDIA GeForce GTX 1080 Ti
✓ Compute capability: 6.1 (sm_61)
✓ Peak VRAM usage: ~3.5 GB
```

---

## ⚠️ DO NOT DO THIS

❌ `pip install --upgrade torch`
- Will downgrade to latest (2.11.0) which doesn't support sm_61
- Falls back to CPU silently
- Training becomes 100x slower

❌ `pip install pytorch::pytorch`
- Conda solver picks incompatible version
- Same sm_61 compatibility issue

❌ Using `requirements-latest.txt` or updating versions
- Breaking change: sm_61 support removed in PyTorch 2.1

---

## 📋 Troubleshooting

### Issue: Running on CPU despite GPU available
```
CUDA initialized but cannot execute kernels on this GPU; falling back to CPU
```

**Fix**:
```bash
pip list | grep torch
# If version > 2.0.1, reinstall:
pip uninstall torch torchvision torchaudio -y
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

### Issue: ImportError when importing transformers
**Fix**: Ensure you're in the venv:
```bash
source venv/bin/activate
python -c "import transformers; print(transformers.__version__)"
```

### Issue: CUDA error in SLURM job
Check `/home2/sankalp0109/src_IS/venv/bin/python` path in SLURM script matches your venv.

---

## 📦 Dependency Lock Rationale

This project uses **PyTorch 2.0.1 specifically** because:

1. **sm_61 Support**: GTX 1080 Ti requires PyTorch 2.0.x or earlier
2. **Stability**: Version 2.0.1 is mature and widely tested for this architecture
3. **CUDA 12.8**: Compatible via `cu121` wheel index
4. **All Downstream Packages**: Locked to versions tested with PyTorch 2.0.1

**Timeline**:
- PyTorch 2.0.x: ✅ Supports sm_61
- PyTorch 2.1+: ❌ Dropped sm_61 support
- PyTorch 2.2+: ❌ sm_61 still not supported

---

## 🔄 For Future Developers

If you need to update dependencies:
1. **Test on GTX 1080 Ti FIRST** before committing
2. Run `check_vram_calibration.py` to verify GPU usage
3. Update `requirements.txt` AND this file in `CLAUDE.md`
4. Document the reason for any version change

**Never**:
- Update `torch==*` in requirements.txt without testing on sm_61
- Use `pip install --upgrade` (always specify exact versions)
- Assume newer PyTorch versions are compatible (they're not for sm_61)

---

## 📚 References

- [PyTorch CUDA Architecture Support](https://pytorch.org/get-started/locally/)
- [GPU Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [GTX 1080 Ti Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/10-series/titan-x/) → Compute Capability: 6.1

---

## 🔢 NumPy Compatibility (PyTorch 2.0.1)

### Critical Issue
PyTorch 2.0.1 was compiled with NumPy 1.x, but NumPy 2.x is incompatible.

**Error if you have NumPy 2.x**:
```
Failed to initialize NumPy: _ARRAY_API not found
```

### Solution
Pin NumPy to version < 2.0:
```
numpy<2
```

This is already in `requirements.txt` - just reinstall:
```bash
pip install 'numpy<2'
# or
pip install -r requirements.txt
```

### Version Matrix
| NumPy | PyTorch 2.0.1 | Status |
|-------|---------------|--------|
| 1.24.x | ✅ Compatible | Use this |
| 2.x | ❌ Incompatible | Will crash |

**Do NOT upgrade NumPy** - keep it at 1.x while using PyTorch 2.0.1.

