import logging
import numpy as np
import os
import sys
import torch

from typing import Any, List, Optional, Tuple

from pytext.trainers import Trainer
from pytext.common.constants import Stage
from pytext.config import PyTextConfig
from pytext.config.pytext_config import ConfigBase
from pytext.data.data_handler import BatchIterator
from pytext.metric_reporters import MetricReporter
from pytext.models.distributed_model import DistributedModel
from pytext.models.model import Model
from pytext.optimizer import learning_rates, optimizer_step, optimizer_zero_grad
from pytext.utils import cuda_utils

from mldc.serialization import ModelState


LOG = logging.getLogger("mldc.HREDTrainer")


class HREDTrainer(Trainer):
  """
  Trains HRED with the option for a variable number of dialogue turns.
  This allows for curriculum learning (e.g. https://arxiv.org/abs/1611.06204).

  Attributes:
  - length_schedule_per_epoch (List[Tuple[int, int]]): list of (epoch_id, dlg length) describing at what epoch to change length
    to.  Epochs must be increasing.

  Notes:
  - The curriculum has the property that each epoch has the same number of datapoints, unlike one-pass and baby-steps.
    It can briefly be described as followed:
      For a sample dialogue `dlg`:
        if len(dlg.turns) >= n_turns @ epoch:
          choose a slice of dlg.turns with length n_turns from the tail of the dialogue
        else:
          choose a slice of dlg.turns with length random.choice([2, 4, ..., len(dlg.turns) ]) from the tail of the dialogue
  - We are always predicting the last turn of the dialogue.
  - The HREDTrainer sets the `n_turns` member on the appropriate Field through the Dataset, so that the Field
    subsamples the dialogue on the `postprocessing` call.

  """

  class Config(ConfigBase):
    # Manual random seed
    random_seed: int = 0
    # Overload if continuing training
    start_epoch: int = 1
    # Training epochs
    epochs: int = 10
    # Stop after how many epochs when the eval metric is not improving
    early_stop_after: int = 0
    # Clip gradient norm if set
    max_clip_norm: Optional[float] = None
    # Whether metrics on training data should be computed and reported.
    report_train_metrics: bool = True
    # List of (epoch, num turns) pairs, in order of ascending epoch
    length_schedule_per_epoch: List[Tuple[int, int]] = [(1, 2)]

  @classmethod
  def from_config(cls, config, *args, **kwargs):
    self = super().from_config(config, *args, **kwargs)
    np.random.RandomState(seed=config.random_seed)
    torch.manual_seed(seed=config.random_seed)
    return self

  def train(
      self,
      train_iter: BatchIterator,
      eval_iter: BatchIterator,
      model: Model,
      metric_reporter: MetricReporter,
      train_config: PyTextConfig,
      optimizers: List[torch.optim.Optimizer],
      scheduler=None,
      rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:

    if cuda_utils.CUDA_ENABLED:
      model = model.cuda()
      if cuda_utils.DISTRIBUTED_WORLD_SIZE > 1:
        device_id = torch.cuda.current_device()
        model = DistributedModel(
            module=model,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
        )

    best_metric = None
    last_best_epoch = 0
    best_model_path = None
    scheduler = self._prepare_scheduler(train_iter, scheduler)

    def training_pre_batch_callback():
      optimizer_zero_grad(optimizers)

    def training_backprop(loss):
      loss.backward()
      if scheduler:
        scheduler.step_batch()

      if self.config.max_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.max_clip_norm
        )
      else:
        grad_norm = None

      optimizer_step(optimizers)
      # grad_norm could be used to check grads sync in distributed training
      return grad_norm

    len_sched_ix = 0

    # Used since we need the infinite iterator (only created and called once)
    def batch_generator_for_epoch(it):
      n = len(it)
      while n > 0:
        yield next(it)
        n -= 1

    for epoch in range(self.config.start_epoch, self.config.epochs + 1):
      # Set the dialogue length in the fields, to be used by the postprocessor
      while self.config.length_schedule_per_epoch \
              and len_sched_ix < len(self.config.length_schedule_per_epoch) \
              and epoch >= self.config.length_schedule_per_epoch[len_sched_ix][0]:
        train_iter.max_n_turns = \
            self.config.length_schedule_per_epoch[len_sched_ix][1]
        eval_iter.max_n_turns = \
            self.config.length_schedule_per_epoch[len_sched_ix][1]
        len_sched_ix += 1

      LOG.info(f"\nRank {rank} worker: Starting epoch #{epoch}")
      model.train()
      lrs = (str(lr) for lr in learning_rates(optimizers))
      LOG.info(f"Learning rate(s): {', '.join(lrs)}")
      self._run_epoch(
          Stage.TRAIN,
          epoch,
          batch_generator_for_epoch(train_iter),
          model,
          metric_reporter,
          pre_batch=training_pre_batch_callback,
          backprop=training_backprop,
          rank=rank,
      )
      model.eval(Stage.EVAL)
      with torch.no_grad():
        eval_metric = self._run_epoch(
            Stage.EVAL, epoch, batch_generator_for_epoch(eval_iter), model, metric_reporter, rank=rank
        )
      # Step the learning rate scheduler(s)
      if scheduler:
        assert eval_metric is not None
        scheduler.step(
            metrics=metric_reporter.get_model_select_metric(eval_metric),
            epoch=epoch,
        )

      # choose best model.
      if metric_reporter.compare_metric(eval_metric, best_metric):
        LOG.info(
            f"Rank {rank} worker: Found a better model! Saving the model state for epoch #{epoch}."
        )
        last_best_epoch = epoch
        best_metric = eval_metric
        # Only rank = 0 trainer saves modules.
        if train_config.save_module_checkpoints and rank == 0:
          best_model_path = os.path.join(
              train_config.modules_save_dir, "best_model"
          )
          optimizer, = optimizers  # PyText only ever returns a single optimizer in this list
          torch.save(ModelState(
            epoch=epoch,
            parameters=model.state_dict(),
            optimizer=optimizer.state_dict(),
          ), best_model_path)

      if (self.config.early_stop_after > 0 and (epoch - last_best_epoch == self.config.early_stop_after)):
        LOG.info(
            f"Rank {rank} worker: Eval metric hasn't changed for "
            f"{self.config.early_stop_after} epochs. Stopping now."
        )
        break
      sys.stdout.flush()

    train_iter.close()
    eval_iter.close()
    model.load_state_dict(torch.load(best_model_path).parameters)
    return model, best_metric
