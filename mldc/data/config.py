from typing import List
from pytext.config.field_config import WordFeatConfig
from pytext.config.field_config import ConfigBase
from pytext.config.module_config import ModuleConfig


class ModelInput:
  SEQ = "seq_word_feat"
  SEQ_EMB = "pretrained_model_embedding"
  TASK_ID = "task_id"
  DOMAIN_ID = "domain_id"
  DLG_ID = "dlg_id"
  DLG_LEN = "dlg_len"


class ModelOutput:
  TOK = "out_tokens"


class ModelInputConfig(ModuleConfig):
  seq_word_feat: WordFeatConfig = WordFeatConfig(
    min_freq=1,
  )
  # pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None


class ModelOutputConfig(ConfigBase):
  _name = ModelOutput.TOK
  export_output_names: List[str] = ['word_scores']
  min_freq: int = 1
