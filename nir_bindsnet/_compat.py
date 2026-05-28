"""Compatibility shim for BindsNET with newer PyTorch versions.

BindsNET 0.2.x references ``torch._six`` which was removed in recent
PyTorch releases.  This module patches the missing symbols so that
BindsNET can be imported without errors.
"""

import collections.abc
import sys
import types


def _patch_torch_six():
    """Inject a ``torch._six`` stub if it does not already exist."""
    if "torch._six" not in sys.modules:
        import torch  # noqa: F401 – ensures torch is initialised first

        module = types.ModuleType("torch._six")
        module.container_abcs = collections.abc
        module.string_classes = (str,)
        module.int_classes = (int,)
        sys.modules["torch._six"] = module


# Apply patch on import so downstream code can ``import bindsnet`` safely.
_patch_torch_six()
