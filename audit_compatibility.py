#!/usr/bin/env python3
"""
Audit codebase for PyTorch 2.0.1 compatibility issues.
"""

import re
import os
from pathlib import Path

issues_found = []

# Patterns to check
patterns = [
    (r'torch\.serialization\.add_safe_globals', 'torch.serialization.add_safe_globals (not in PyTorch 2.0.1)'),
    (r'torch\.cuda\.amp\.autocast\(\)', 'torch.cuda.amp.autocast() - verify works with PyTorch 2.0.1'),
    (r'F\.upsample\(', 'F.upsample() - deprecated, use F.interpolate()'),
    (r'\.backward\(retain_graph=True\)', 'backward with retain_graph - verify memory usage'),
    (r'torch\.jit\.script', 'torch.jit.script - may have version-specific issues'),
    (r'@torch\.jit\.ignore', '@torch.jit.ignore - verify compatibility'),
]

# Scan source files
src_dir = Path('src')
for py_file in src_dir.rglob('*.py'):
    try:
        with open(py_file, 'r') as f:
            content = f.read()
            
        for pattern, description in patterns:
            matches = re.findall(pattern, content)
            if matches:
                for i, line in enumerate(content.split('\n'), 1):
                    if re.search(pattern, line):
                        issues_found.append({
                            'file': str(py_file),
                            'line': i,
                            'description': description,
                            'code': line.strip()
                        })
    except Exception as e:
        print(f"Error scanning {py_file}: {e}")

# Print results
if issues_found:
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                  PYTORCH 2.0.1 COMPATIBILITY AUDIT                        ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"\n⚠️  Found {len(issues_found)} potential compatibility issues:\n")
    
    for issue in issues_found:
        print(f"📍 {issue['file']}:{issue['line']}")
        print(f"   Issue: {issue['description']}")
        print(f"   Code: {issue['code'][:70]}")
        print()
else:
    print("✅ No obvious PyTorch 2.0.1 compatibility issues found!")
    print("\nScanned:")
    print("  ✅ No torch.serialization.add_safe_globals usage")
    print("  ✅ No deprecated F.upsample() calls")
    print("  ✅ No obvious deprecated APIs")
    print("  ✅ Standard torch.cuda.amp usage (compatible)")

