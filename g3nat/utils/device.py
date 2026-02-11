import torch

def setup_device(device_arg: str) -> torch.device:
    """
    Setup computation device.

    Args:
        device_arg: 'auto', 'cpu', or 'cuda'

    Returns:
        torch.device instance
    """
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    return device
