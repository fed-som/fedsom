import torch


def assign_device(n: int, gpu_count: int) -> torch.device:

    return torch.device(f"cuda:{n%gpu_count}") if torch.cuda.is_available() else torch.device("cpu")
