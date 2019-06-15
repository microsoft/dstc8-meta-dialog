from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ModelState:
  epoch: int = 0
  parameters: Optional[Union[dict, OrderedDict]] = None
  optimizer: Optional[Union[dict, OrderedDict]] = None
  lr_scheduler: Optional[Union[dict, OrderedDict]] = None
  task_config: Optional[dict] = None
