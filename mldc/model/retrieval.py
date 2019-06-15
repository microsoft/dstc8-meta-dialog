import torch
from pytext.config import ConfigBase
from pytext.models.embeddings import EmbeddingBase, EmbeddingList
from pytext.models.model import Model
from pytext.models.representations.representation_base import RepresentationBase
from pytext.models.decoders import DecoderBase
from pytext.models.output_layers import OutputLayerBase
from typing import Optional, List, Dict, Tuple, Any


class RetrievalOutputLayer(OutputLayerBase):
  class Config(ConfigBase):
    pass

  @classmethod
  def from_config(cls, config: Config, meta: Any):
    return cls(config)


class RetrievalRepresentation(RepresentationBase):
  class Config(RepresentationBase.Config):
    pass

  def __init__(self, config: Config, embed_dim: int) -> None:
    super().__init__(config)
    self.representation_dim = embed_dim

  def forward(
      self,
      seq_embed: torch.Tensor,
      seq_lengths: torch.Tensor,
      word_lengths: torch.Tensor,
      *args,
  ) -> torch.Tensor:

    for dlg_i, dlg_n_turns in enumerate(seq_lengths):
      for turn_i, n_words in zip(range(dlg_n_turns), word_lengths[dlg_i]):
        seq_embed[dlg_i, turn_i, 0] = seq_embed[dlg_i, turn_i, :n_words].sum(dim=0)
    seq_embed = seq_embed[:, :, 0]
    seq_embed = torch.cumsum(seq_embed, dim=1)

    return seq_embed


class RetrievalDecoder(DecoderBase):
  class Config(ConfigBase):
    pass

  @classmethod
  def from_config(cls, config: Config, in_dim: int, out_dim: int):
    return cls(config)

  def forward(self, dlg_states: Tuple[torch.Tensor, ...],
              encoder_inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
    pass


class RetrievalModel(Model):
  class Config(ConfigBase):
    representation: RetrievalRepresentation.Config = RetrievalRepresentation.Config()
    decoder: RetrievalDecoder.Config = RetrievalDecoder.Config()
    output_layer: RetrievalOutputLayer.Config = RetrievalOutputLayer.Config()

  @staticmethod
  def distance(a, b):
    """
    calculate the distances of each element in b to all elements in a

    args:
      a: tensor (n, feature-dim)
      b: tensor (m, feature-dim)

    returns: tensor (n, m)
    """
    # Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
    #                          = dot(u / norm(u), v / norm(v))
    # We fist normalize the rows, before computing their dot products via transposition:
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.einsum('btd,id->bti', a_norm, b_norm)

  def get_loss(self, logit, target, context):
    # TODO what's a good loss to report here? Or maybe, what should "logit" be in the first place?
    return torch.zeros((1,), dtype=torch.float32)

  def forward(self, seqs: torch.Tensor, token_emb: torch.Tensor, tf_inputs: Optional[torch.Tensor],
              n_seqs: torch.Tensor, n_seq_words: torch.Tensor,
              n_tf_words: Optional[torch.Tensor], *,
              responses: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> List[torch.Tensor]:
    rep = self.representation(token_emb, n_seqs, n_seq_words)

    if self.training:
      assert responses is not None
      # training: save bag of embeddings and corresponding outputs
      # TODO allow accumulation
      rep = rep[:, ::2]  # every second input is a prompt, the others are responses
      self._rep = rep.reshape(rep.shape[0] * rep.shape[1], rep.shape[2])
      resp, resp_lens = responses
      assert len(resp) == len(resp_lens)
      assert len(resp) == rep.shape[0]
      self._resp = resp.reshape(rep.shape[0] * rep.shape[1], -1)
      self._resp_lens = resp_lens.reshape(rep.shape[0] * rep.shape[1])
      return [torch.zeros((1,), dtype=torch.float32, device=rep.data.device)]

    # evaluation: in saved state, retrieve most similar response
    # find the response who's state is most similar to the last turn's state
    dist = self.distance(rep[:, -1:], self._rep)   # B,1,Memsize
    dist_min = torch.argmax(dist, dim=-1)  # B,1
    res = self._resp[torch.flatten(dist_min)]
    return res.reshape(rep.shape[0], 1, -1), self._resp_lens[torch.flatten(dist_min)]

  @classmethod
  def compose_embedding(
      cls, sub_emb_module_dict: Dict[str, EmbeddingBase]
  ) -> EmbeddingList:
    """disable concatenation of input and output sequences"""
    # return EmbeddingList(sub_emb_module_dict.values(), concat=False)
    seq, *_ = sub_emb_module_dict.values()
    return EmbeddingList((seq,), concat=False)
