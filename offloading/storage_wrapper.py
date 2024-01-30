"""Wrapper for a module that puts all tensors into a shared storage; based on original work by dvmazur@ and lavawolfiee@"""
from typing import Tuple
import torch
import torch.nn as nn


class ModuleWithStorage(nn.Module):
    """
    Wraps a module and puts all its parameters and buffers to a shared storage (torch.UntypedStorage).
    WARNING: this wrapper modifies the input module in-place so that it can no longer change device or be trained.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.storage, self.module = self.put_on_storage_inplace_(module)

    @staticmethod
    def put_on_storage_inplace_(module: nn.Module) -> Tuple[torch.UntypedStorage, nn.Module]:
        """Modify module so that every parameter and buffer is a pointer to a pre-allocated storage"""
        device = next(module.parameters()).device
        storage_size_bytes = 0
        offsets = [0]
        module_data = dict()
        module_data.update(module.named_parameters())
        module_data.update(module.named_buffers())

        for x in module_data.values():
            assert isinstance(x, torch.Tensor)
            storage_size_bytes += x.nbytes
            offsets.append(storage_size_bytes)

        storage = torch.UntypedStorage(storage_size_bytes, device=device)

        i = 0
        for x in module_data.values():
            assert isinstance(x, torch.Tensor)
            start, end = offsets[i], offsets[i + 1]
            storage_view = torch.as_tensor(
                storage[start: end], dtype=x.dtype, device=device).view(x.shape)
            storage_view.copy_(x)
            assert storage_view.data_ptr() == storage.data_ptr() + start
            x.data = storage_view  # <-- replace parameter/buffer with a pointer to storage
            i += 1

        for k, v in module.state_dict().items():
            assert storage.data_ptr() <= v.data_ptr() <= storage.data_ptr() + storage.nbytes(), k
        return storage, module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

