import logging
import numpy as np
import sys

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, islice

from mldc.data.schema import MetaDlgDataDialog
from pytext.config.component import Component, ComponentType, ConfigBase
from typing import Sequence
from tqdm import tqdm

from mldc.preprocessing.input_embedding import EmbedderInterface
from mldc.util import TqdmToLogger


LOG = logging.getLogger('mldc.preprocessing.featurizer')
INIT_DONE = False


def no_progress():
  """
  determines if we want to see progressbars in the output

  do not show progress bars if:
  - if we aren't on an interactive terminal or
  - the user wants verbose logging
  """
  return False
  return not sys.stdout.isatty()


class OutputRecord:
  turns: Sequence[str]
  token_ids: np.array
  domain_id: str
  task_id: str
  dlg_id: str


class TokenIdFeaturizer(Component):
  __COMPONENT_TYPE__ = ComponentType.FEATURIZER
  __EXPANSIBLE__ = True
  """
  This "featurizes" the whole dataset. Because the data is quite big and needs to fit in RAM,
  the featurizer only tokenizes the text converts tokens to integer IDs, thereby compressing
  it from the original text format.

  The action can be performed in parallel by a pool of workers.
  """

  class Config(Component.Config):
    pass

  @classmethod
  def from_config(cls, config: Config, feature_config: ConfigBase, text_embedder_config: EmbedderInterface.Config):
    return cls(config, feature_config, text_embedder_config)

  def __init__(self, config: Config, feature_config, text_embedder_config):
    self.text_embedder = EmbedderInterface.from_config(text_embedder_config)

  def featurize_batch(self, input_record_list: Sequence[MetaDlgDataDialog]) -> Sequence[OutputRecord]:
    return [self.featurize(record) for record in input_record_list]

  def featurize(self, input_record: MetaDlgDataDialog):
    ret = OutputRecord()
    ret.token_ids = [
      np.array([self.text_embedder.bos_idx] + self.text_embedder.encode_text_as_ids(turn).tolist() + [self.text_embedder.eos_idx])
      for turn in input_record.turns]
    ret.task_id = input_record.task_id
    ret.domain_id = input_record.domain
    ret.dlg_id = input_record.id
    return ret

  @classmethod
  def _featurize_worker(
    cls,
    config,
    text_embedder_cfg,
    feature_config,
    batch: Sequence[MetaDlgDataDialog],
  ) -> Sequence[OutputRecord]:
    # init/initargs isn't supported in python 3.6 yet, so we emulate it here
    global FEATURIZER, INIT_DONE
    if not INIT_DONE:
      FEATURIZER = cls(config, text_embedder_config=text_embedder_cfg, feature_config=feature_config)
      INIT_DONE = True
    if not len(batch):
      return []
    return FEATURIZER.featurize_batch(batch)

  @classmethod
  def parallel_featurize_batch(
    cls,
    batch: Sequence[MetaDlgDataDialog],
    max_workers=4,
    chunksize: int = 1000,
    text_embedder_cfg: EmbedderInterface.Config = None,
    feature_config=None
  ) -> Sequence[OutputRecord]:
    # tokenizer models are relatively small so load separately in each process
    config = TokenIdFeaturizer.Config()

    # function to split the input iterator into smaller ones, one for each process
    def split_iterator(iterator, chunksize=chunksize):
      iterator = iter(iterator)
      while True:
        chunk = tuple(islice(iterator, chunksize))
        if not chunk:
          return
        yield chunk

    worker = partial(cls._featurize_worker, config, text_embedder_cfg, feature_config)

    LOG.debug("featurizing data with %d workers", max_workers)
    tqdm_out = TqdmToLogger(LOG, level=logging.INFO)
    if max_workers > 1:
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
        featurized_lol, futures = [], []
        with tqdm(unit=" chunks", desc="Loading dialogues in chunks of size %d" % chunksize, disable=no_progress(),
                  file=tqdm_out, mininterval=30) as bar:
          chunks = iter(split_iterator(batch))
          for chunk in chunks:
            bar.update(1)
            futures += executor.submit(worker, chunk),
        for future in tqdm(futures, unit=" chunks", desc="Processing dialogues in chunks of size %d" % chunksize,
                           disable=no_progress(), file=tqdm_out, mininterval=30):
            featurized_lol.append(future.result())
    else:
      featurized_lol = [worker(b) for b in split_iterator(batch)]
      # clear the INIT flag for next time this is called
      global INIT_DONE
      INIT_DONE = False

    # merge the results back into a single iterator
    return chain(*featurized_lol)
