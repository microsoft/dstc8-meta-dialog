import torch
import logging
import os
from collections import defaultdict
from typing import Dict
from pytext.config.pytext_config import ConfigBase
from pytext.metric_reporters import MetricReporter, channel as C
from mldc.preprocessing.input_embedding import EmbedderInterface
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from runstats.fast import Statistics


LOG = logging.getLogger("mldc.metrics")


class TaskMetric:
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def _asdict(self):
    """pretend to be a ConfigBase object"""
    return self.kwargs

  def __str__(self):
    return ' '.join([f'{k}={v:2.4f}' for k, v in self._asdict().items()])


class TaskMetrics:
  tasks: Dict[str, TaskMetric] = dict()
  all: TaskMetric = TaskMetric()

  def _asdict(self):
    """pretend to be a ConfigBase object"""
    res = {'all': self.all}
    for task, tm in self.tasks.items():
      res['tasks/' + task] = tm
    return res

  def __str__(self):
    return str(self.all)


class MetaLearnMetricReporter(MetricReporter):
  class Config(ConfigBase):
    output_path: str = '.'

  def __init__(self, config, metadata, text_embedder: EmbedderInterface.Config):
    super().__init__(channels=[
      C.TensorBoardChannel(SummaryWriter(os.path.join(config.output_path, 'logs'))),
      C.ConsoleChannel()])
    self.text_embedder = EmbedderInterface.from_config(text_embedder)
    self._reset()

  def _reset(self):
    super()._reset()
    self._tasks = set()
    # stats[accumulator_name][task] : Statistics
    self.stats = defaultdict(lambda: defaultdict(Statistics))
    self.predictions = defaultdict(list)
    self.references = defaultdict(list)

  def to_words(self, descr, ids: torch.Tensor):
    # use trivial tokenizer
    text = self.text_embedder.decode_ids_as_text(ids.tolist())
    LOG.debug("%s: %s", descr, text)
    return text.split()

  def _add_batch_predictions(self, task, pred, tgt):
    pred, pred_lens = pred
    tgt, tgt_lens = tgt

    pred = pred.reshape(-1, pred.shape[-1])
    tgt = tgt.reshape(-1, tgt.shape[-1])
    pred_lens = pred_lens.flatten()
    tgt_lens = tgt_lens.flatten()

    self.predictions[task].extend([self.to_words('PP', s[:pl]) for s, pl, tl in zip(pred, pred_lens, tgt_lens) if tl > 0])
    self.references[task].extend([[self.to_words('TT', s[:tl])] for s, tl in zip(tgt, tgt_lens) if tl > 0])

  def add_batch_stats(self, task, t_loss, s_inputs, t_predictions=None, t_targets=None, **kwargs):
    task = task.replace('dialogues/', '').replace('.txt', '')
    self._tasks.add(task)
    if t_predictions is not None and t_targets is not None:
      self._add_batch_predictions(task, t_predictions, t_targets)
    self.stats['n_turns'][task].push(s_inputs[0].shape[1])
    self.stats['t_loss'][task].push(t_loss)
    for k, v in kwargs.items():
      self.stats[k][task].push(v)
    self.all_loss.append(t_loss)

  def overlap_metrics(self, references, predictions):
    for k, v in dict(bleu1=[1.],
                     bleu2=[0.5, 0.5],
                     bleu3=[0.33, 0.33, 0.33],
                     bleu4=[0.25, 0.25, 0.25]).items():
      yield k, corpus_bleu(references, predictions, weights=v)

  def all_task_metric(self):
    def ssum(s):
      # note: sum does not take keyword arguments
      return sum(s, Statistics())

    kwargs = {k: ssum(task_stats.values()).mean() for k, task_stats in self.stats.items()}
    if sum(len(p) for p in self.predictions.values()):
      kwargs.update(self.overlap_metrics(sum([self.references[t] for t in self._tasks], []),
                                         sum([self.predictions[t] for t in self._tasks], [])))

    return TaskMetric(
      n_updates=len(ssum(self.stats['n_turns'].values())),
      **kwargs)

  def calculate_metric(self):
    res = TaskMetrics()
    for task in self._tasks:
      kwargs = {k: self.stats[k][task].mean() for k in self.stats}
      if len(self.predictions[task]):
        kwargs.update(self.overlap_metrics(self.references[task], self.predictions[task]))
      res.tasks[task] = TaskMetric(n_updates=len(self.stats['n_turns']),
                                   **kwargs)
    res.all = self.all_task_metric()
    return res
