import json

from io import TextIOWrapper
from mldc.data.schema import MetaDlgDataDialog
from typing import Iterable
from zipfile import ZipFile


def stream_dlgs(zipfile: str, file_in_zip: str, loop: bool = False, max_turns: int = -1) -> Iterable[MetaDlgDataDialog]:
  with ZipFile(zipfile) as zf:
    while True:
      with zf.open(file_in_zip, 'r') as f:
        for line in TextIOWrapper(f, encoding='utf-8'):
          dlg = MetaDlgDataDialog(**json.loads(line))
          if max_turns > 1:
            dlg.turns = dlg.turns[:max_turns]
          yield dlg
      if not loop:
        break


def stream_text(zipfile: str, file_in_zip: str,
                loop: bool = False, lower: bool = False, max_turns: int = -1) -> Iterable[str]:
  for dlg in stream_dlgs(zipfile, file_in_zip, loop, max_turns):
    for turn in dlg.turns:
      if lower:
        yield turn.lower()
      else:
        yield turn
