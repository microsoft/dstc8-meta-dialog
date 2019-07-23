import io
import click
import re
import os
import json
import itertools
import logging
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
    if out_dir is None:
      return
    domain = self._domain_name(domain)
    os.makedirs(os.path.join(out_dir, domain), exist_ok=True)
    self.json_file = open(os.path.join(out_dir, domain, "results.jsonl"), "w")
    self.cnt = 0

  @staticmethod
  def _domain_name(domain):
    return os.path.splitext(os.path.basename(domain))[0]

  def __enter__(self):
    return self

  def add(self, target, prediction, dlg_id, predict_turn):
    if self.json_file is None:
      return
    self.cnt += 1
    prediction = re.sub(r'\s+', ' ', prediction)
    target = re.sub(r'\s+', ' ', target)
    json.dump(dict(dlg_id=dlg_id, predict_turn=predict_turn, response=prediction),
              self.json_file)
    self.json_file.write("\n")

  def __exit__(self, type, value, traceback):
    if self.json_file is None:
      return
    self.json_file.close()


def overwrite_param(dct: Dict, key: str, value: Any):
  assert key in dct, "Unknown configuration parameter %s" % key
  dct[key] = value


def output_dir_option(f):
  def callback(ctx, param, value):
    os.environ['MLDC_OUTPUT_DIR'] = value
    return value
  return click.option('--output-dir', type=click.Path(file_okay=False), expose_value=False,
                      default=lambda: os.environ.get('MLDC_OUTPUT_DIR', "exp/debug"),
                      help="write outputs to this directory", callback=callback)(f)


def exp_dir(basename=None):
  path = os.environ.get('MLDC_OUTPUT_DIR', 'exp/debug')
  os.makedirs(path, exist_ok=True)
  if basename is None:
    return path
  return os.path.join(path, basename)


def validate_length_schedule(ctx, param, value):
  try:
    parsed = json.loads(value)
    if not all([t % 2 == 0 for e, t in parsed]):
      raise RuntimeError('Number of turns are not all even!')
    if not all([a == b for a, b in zip([e for e, t in parsed], sorted([e for e, t in parsed]))]):
      raise RuntimeError('Epochs are not ascending!')
    if parsed[0][0] != 1:
      raise RuntimeError('First epoch is not 1 in the length schedule!')
    return parsed
  except Exception as e:
    raise click.BadParameter('Could not parse length schedule! %s' % str(e))


class TqdmToLogger(io.StringIO):
  """
  Output stream for TQDM which will output to logger module instead of
  the StdOut.
  """
  logger = None
  level = None
  buf = ''

  def __init__(self, logger, level=None):
    super().__init__()
    self.logger = logger
    self.level = level or logging.INFO

  def write(self, buf):
    self.buf = buf.strip('\r\n\t ')

  def flush(self):
    self.logger.log(self.level, self.buf)
