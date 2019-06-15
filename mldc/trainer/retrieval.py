import os
import torch
import logging

from typing import Any, Optional, Tuple

from pytext.trainers import Trainer
from pytext.config import PyTextConfig
from pytext.config.pytext_config import ConfigBase
from pytext.common.constants import Stage
from pytext.models.model import Model
from pytext.utils import cuda_utils

from mldc.data.data_handler import BatchPreparationPipeline
from mldc.metrics.metrics import MetaLearnMetricReporter


TASKS_AGGR = 0
SUPPORT_ON_SLOW = 1
TARGET_ON_FAST = 2

EPSILON = 0.001
LOG = logging.getLogger("mldc.trainer")


class RetrievalTrainer(Trainer):

  class Config(ConfigBase):
    random_seed: int = 0
    # Whether metrics on training data should be computed and reported.
    report_train_metrics: bool = True

  def test(self, test_task_iters: BatchPreparationPipeline,
           model: Model,
           metric_reporter: MetaLearnMetricReporter):

    for mbidx, meta_batch in enumerate(test_task_iters):
      support, target, context = meta_batch
      for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
        task = t_context['task_id'][0]
        model.train()
        model.contextualize(s_context)
        model(*s_inputs, responses=s_targets)  # model remembers responses
        model.eval()

        with torch.no_grad():
          t_pred = model(*t_inputs)
          t_loss = model.get_loss(t_pred, t_targets, t_context).item()

          metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                          t_predictions=t_pred, t_targets=t_targets)

    metric_reporter.report_metric(stage=Stage.TEST, epoch=0, reset=False)

  def predict(self, test_task_iters: BatchPreparationPipeline,
              model: Model,
              metric_reporter: MetaLearnMetricReporter):

    for meta_batch in test_task_iters:
      support, target, context = meta_batch
      for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
        task = t_context['task_id'][0]
        model.train()
        model.contextualize(s_context)
        model(*s_inputs, responses=s_targets)  # model remembers responses
        model.eval()

        with torch.no_grad():
          resps, resp_lens = model(*t_inputs)

          yield dict(task=task, resps=resps, resp_lens=resp_lens,
                     s_inputs=s_inputs, s_targets=s_targets, s_context=s_context,
                     t_inputs=t_inputs, t_targets=t_targets, t_context=t_context)

  def train(
      self,
      train_task_iters: Optional[BatchPreparationPipeline],
      eval_task_iters: BatchPreparationPipeline,
      model: Model,
      metric_reporter: MetaLearnMetricReporter,
      train_config: PyTextConfig,
      rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:

    if cuda_utils.CUDA_ENABLED:
      model = model.cuda()

    best_model_path = None

    # Start outer loop (meta learner "epochs") #############################################
    if not train_task_iters:
      LOG.warning("Model does not need meta-training")
    else:
      for epoch in range(1, 2):  # single epoch
        for bidx, (support, target, context) in zip(range(100), train_task_iters):
          for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
            task = t_context['task_id'][0]

            # Adapt the model using the support set
            model.train()
            for step in range(1):
              model.contextualize(s_context)
              model(*s_inputs, responses=s_targets)  # model remembers responses

            # Evaluate the model using the target set
            model.eval()    # model now retrieves from examples seen so far
            model.contextualize(t_context)
            t_pred = model(*t_inputs)
            t_loss = model.get_loss(t_pred, t_targets, t_context).item()
            metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                            t_predictions=t_pred, t_targets=t_targets)

        metric_reporter.report_metric(stage=Stage.TRAIN, epoch=epoch, reset=False)

      logging.info("Evaluating model on eval tasks")
      with torch.no_grad():
        for bidx, (support, target, context) in enumerate(eval_task_iters):
          for (s_inputs, t_inputs), (s_targets, t_targets), (s_context, t_context) in zip(support, target, context):
            task = t_context["task_id"][0]
            model.train()
            model.contextualize(s_context)
            model(*s_inputs, responses=s_targets)  # model remembers responses
            model.eval()
            t_pred = model(*t_inputs)
            t_loss = model.get_loss(t_pred, t_targets, t_context).item()

            metric_reporter.add_batch_stats(task, t_loss, s_inputs,
                                            t_predictions=t_pred, t_targets=t_targets)

      metric_reporter.report_metric(stage=Stage.EVAL, epoch=epoch, reset=False)

    best_model_path = os.path.join(
        train_config.modules_save_dir, "model.pt"
    )
    torch.save(model.state_dict(), best_model_path)

    return model, None
