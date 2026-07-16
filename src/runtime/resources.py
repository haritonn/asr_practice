"""Resource lifecycle helpers shared by model adapters."""

import gc

import torch


def release_accelerator_memory() -> None:
    """Release cached accelerator memory after a model has been discarded."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
