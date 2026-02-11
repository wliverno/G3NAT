# tests/test_utils/test_device.py
import sys
sys.path.insert(0, '.')

import torch

def test_setup_device_auto():
    """Test device setup with auto detection."""
    from g3nat.utils.device import setup_device

    device = setup_device('auto')
    assert isinstance(device, torch.device)
    assert device.type in ['cpu', 'cuda']

def test_setup_device_explicit():
    """Test explicit device selection."""
    from g3nat.utils.device import setup_device

    device = setup_device('cpu')
    assert device.type == 'cpu'
