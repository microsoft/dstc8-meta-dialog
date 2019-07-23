from contextlib import ExitStack, closing

from pprint import pprint
from pytext.task import Task

from pytext.config import ConfigBase, config_to_json
from pytext.config.component import (
    create_data_handler,
    create_featurizer,
    create_metric_reporter,
    create_model,
    create_trainer,
)
from pytext.utils import cuda_utils

from mldc.data.data_handler import MetaDataHandler
from mldc.preprocessing.featurizer import TokenIdFeaturizer
from mldc.data.config import ModelOutputConfig, ModelInputConfig
from mldc.model.retrieval import RetrievalModel
from mldc.trainer.retrieval import RetrievalTrainer
from mldc.metrics.metrics import MetaLearnMetricReporter
from mldc.preprocessing.input_embedding import EmbedderInterface


class RetrievalTask(Task):
  class Config(ConfigBase):
    model: RetrievalModel.Config = RetrievalModel.Config()
    trainer: RetrievalTrainer.Config = RetrievalTrainer.Config(
      report_train_metrics=True,
      save_modules_checkpoint=True,
      modules_save_dir="exp/retrieval"
    )
    featurizer: TokenIdFeaturizer.Config = TokenIdFeaturizer.Config()
    features: ModelInputConfig = ModelInputConfig()
    labels: ModelOutputConfig = ModelOutputConfig()   # was: WordLabelConfig
    metric_reporter: MetaLearnMetricReporter.Config = MetaLearnMetricReporter.Config()
    text_embedder: EmbedderInterface.Config = EmbedderInterface.Config()
    # Maybe we could just have a single instance, would need the nested batch iterator
    data_handler: MetaDataHandler.Config = MetaDataHandler.Config()
    model_needs_meta_training: bool = True

  @classmethod
  def from_config(cls, task_config, metadata=None, model_state=None):
    print("Task parameters:\n")
    pprint(config_to_json(type(task_config), task_config))
    featurizer = create_featurizer(
      task_config.featurizer, task_config.features,
      text_embedder_config=task_config.text_embedder)
    # load data
    data_handler = create_data_handler(
      task_config.data_handler,
      task_config.features,
      task_config.labels,
      text_embedder_config=task_config.text_embedder,
      featurizer=featurizer,
    )
    print("\nLoading data...")
    if metadata:
      data_handler.load_metadata(metadata)
    else:
      data_handler.init_metadata()

    metadata = data_handler.metadata
    task_config.features.seq_word_feat.embed_dim = data_handler.text_embedder.embed_dim

    model = create_model(task_config.model, task_config.features, metadata)
    if model_state:
      model.load_state_dict(model_state)
    if cuda_utils.CUDA_ENABLED:
      model = model.cuda()
    metric_reporter = create_metric_reporter(
      task_config.metric_reporter, metadata,
      text_embedder=task_config.text_embedder)

    return cls(
      trainer=create_trainer(task_config.trainer),
      data_handler=data_handler,
      model=model,
      metric_reporter=metric_reporter,
      model_needs_meta_training=task_config.model_needs_meta_training,
    )

  def __init__(
    self,
    trainer: RetrievalTrainer,
    data_handler: MetaDataHandler,
    model: RetrievalModel,
    metric_reporter: MetaLearnMetricReporter,
    model_needs_meta_training: bool
  ) -> None:
    self.trainer: RetrievalTrainer = trainer
    self.data_handler: MetaDataHandler = data_handler
    self.model: RetrievalModel = model
    self.metric_reporter: MetaLearnMetricReporter = metric_reporter
    self.model_needs_meta_training = model_needs_meta_training

  def train(self, train_config, rank=0, world_size=1):
    """
    Wrapper method to train the model using :class:`~Trainer` object.

    Args:
        train_config (PyTextConfig): config for training
        rank (int): for distributed training only, rank of the gpu, default is 0
        world_size (int): for distributed training only, total gpu to use, default
            is 1
    """
    with ExitStack() as stack:
      train_iter = None
      if self.model_needs_meta_training:
        train_iter = self.data_handler.get_train_iter(
          stack.enter_context(closing(train_iter)))
      eval_iter = self.data_handler.get_eval_iter(repeat=False)
      stack.enter_context(closing(eval_iter))
      return self.trainer.train(
          train_iter,
          eval_iter,
          self.model,
          self.metric_reporter,
          train_config,
          rank=rank,
      )

  def predict(self, test_path, meta_batch_spec_file):
    self.data_handler.test_path = test_path
    for result in self.trainer.predict(self.data_handler.get_test_iter(meta_batch_spec_file=meta_batch_spec_file),
                                       self.model,
                                       self.metric_reporter):
      text_responses = [self.data_handler.text_embedder.decode_ids_as_text(turn[:tlen].tolist())
                        for turn, tlen in zip(result['resps'][0], result['resp_lens'])]
      result['text_responses'] = text_responses
      yield result

  def test(self, test_path, meta_batch_spec_file):
    self.data_handler.test_path = test_path
    self.trainer.test(self.data_handler.get_test_iter(meta_batch_spec_file=meta_batch_spec_file),
                      self.model,
                      self.metric_reporter)

  def export(self, model, export_path, summary_writer=None, export_onnx_path=None):
    pass
