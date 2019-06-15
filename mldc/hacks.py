import sys
import torch
from torch.utils.data import dataloader
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)


def disable_shared_memory():
  """
  Sometimes a runtime error occurs when the shared memory in a virtual machine
  is too small to hold an object. This function disables using shared memory.

  https://github.com/huaweicloud/dls-example/issues/26
  """
  setattr(dataloader, 'default_collate', default_collate_override)

  for t in torch._storage_classes:
    if sys.version_info[0] == 2:
      if t in ForkingPickler.dispatch:
          del ForkingPickler.dispatch[t]
    else:
      if t in ForkingPickler._extra_reducers:
          del ForkingPickler._extra_reducers[t]
