""""
defines a class that maps to the JSON input format and can be used with pydantic.
"""
from typing import List, Optional
from pydantic import BaseModel


class MetaDlgDataDialog(BaseModel):
  id: Optional[str]
  domain: str = ""
  task_id: str = ""
  user_id: str = ""
  bot_id: str = ""
  turns: List[str]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class MetaDlgDataDialogList(BaseModel):
  dialogs: List[MetaDlgDataDialog]
