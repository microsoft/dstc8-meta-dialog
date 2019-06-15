import numpy as np

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain, islice, tee
from multiprocessing import cpu_count

from mldc.data.schema import MetaDlgDataDialog
from pytext.config.component import Component, ComponentType, ConfigBase
from typing import Sequence

from mldc.preprocessing.input_embedding import EmbedderInterface


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
    config: Config,
    batch: Sequence[MetaDlgDataDialog],
    text_embedder_cfg,
    feature_config
  ) -> Sequence[OutputRecord]:
    return cls(config, feature_config, text_embedder_cfg).featurize_batch(batch)

  @classmethod
  def parallel_featurize_batch(
    cls,
    batch: Sequence[MetaDlgDataDialog],
    max_workers=cpu_count(),
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
        # TODO: iterator copying might be more expensive than just realizing the whole iterator as a list
        islice_orig, islice_copy = tee(islice(iterator, chunksize))
        if not tuple(islice_copy):
          return
        yield islice_orig

    featurize_func = partial(cls._featurize_worker, config, text_embedder_cfg=text_embedder_cfg,
                             feature_config=feature_config)
    if max_workers > 1:
      with ProcessPoolExecutor(max_workers=max_workers) as executor:
        featurized_lol = executor.map(featurize_func, split_iterator(batch))
    else:
      featurized_lol = [featurize_func(b) for b in split_iterator(batch)]

    # merge the results back into a single iterator
    return chain(*featurized_lol)
