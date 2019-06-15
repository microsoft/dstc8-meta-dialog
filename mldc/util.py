import re
import os
import json
import itertools
from collections import deque
from typing import Dict, Any


# from https://stackoverflow.com/questions/3345785/getting-number-of-elements-in-an-iterator-in-python
def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


class NLGEvalOutput:
  def __init__(self, out_dir, domain):
    self.hyp_file = None
    if out_dir is None:
      return
    domain = self._domain_name(domain)
    os.makedirs(os.path.join(out_dir, domain), exist_ok=True)
    self.hyp_file = open(os.path.join(out_dir, domain, "hyp.txt"), "w")
    self.ref_file = open(os.path.join(out_dir, domain, "ref.txt"), "w")
    self.json_file = open(os.path.join(out_dir, domain, "results.jsonl"), "w")
    self.cnt = 0

  @staticmethod
  def _domain_name(domain):
    return os.path.splitext(os.path.basename(domain))[0]

  def __enter__(self):
    return self

  def add(self, target, prediction, dlg_id, predict_turn):
    if self.hyp_file is None:
      return
    self.cnt += 1
    prediction = re.sub(r'\s+', ' ', prediction)
    target = re.sub(r'\s+', ' ', target)
    self.hyp_file.write(prediction + "\n")
    self.ref_file.write(target + "\n")
    self.json_file.write(json.dumps(
      dict(dlg_id=dlg_id, predict_turn=predict_turn, response=prediction)))

  def __exit__(self, type, value, traceback):
    if self.hyp_file is None:
      return

    self.hyp_file.close()
    self.ref_file.close()
    self.json_file.close()


def overwrite_param(dct: Dict, key: str, value: Any):
  assert key in dct, "Unknown configuration parameter %s" % key
  dct[key] = value
