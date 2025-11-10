#!/usr/bin/env python3
"""
Evaluation script that works around PyTorch 2.6 compatibility issues
"""

import sys
import torch
from pathlib import Path

# Workaround for PyTorch 2.6 compatibility
import torch.serialization
original_load = torch.load

def patched_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False for backward compatibility"""
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)

torch.load = patched_load

# Now import nerfstudio
from nerfstudio.scripts.eval import ComputePSNR
import tyro


def main():
    """Run evaluation with PyTorch 2.6 compatibility fix"""
    print("Running evaluation with PyTorch 2.6 compatibility patch...")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    sys.exit(main())
