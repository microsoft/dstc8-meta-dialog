import logging
import json

from io import TextIOWrapper
from mldc.data.schema import MetaDlgDataDialog
from typing import Iterable
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count


LOG = logging.getLogger("mldc.preprocessing.stream")


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


def worker(zipfile, file_in_zip, max_turns):
  return list(stream_dlgs(zipfile, file_in_zip, max_turns=max_turns))


def stream_dlgs_many(zipfile: str, files_in_zip: str,
                     max_turns: int = -1, min_domain_size=None) -> Iterable[MetaDlgDataDialog]:

  fn = partial(worker, zipfile, max_turns=max_turns)
  with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    futures = []
    for file in files_in_zip:
      futures.append(executor.submit(fn, file))
    n_skip, n_keep = 0, 0
    for file, fut in zip(files_in_zip, futures):
      res = fut.result()
      if min_domain_size and len(res) < min_domain_size:
        n_skip += 1
        LOG.warning("Skipping domain %s because it is too small (%d < %d), skip/keep: %d/%d",
                    file, len(res), min_domain_size, n_skip, n_keep)
        continue
      n_keep += 1
      yield from res
