import os
import torch
import fasttext
import functools
import numpy as np

import pytext.utils.cuda_utils as cuda_utils

from abc import abstractmethod
from collections import OrderedDict
from sentencepiece import SentencePieceProcessor
from typing import List
from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytext.config.field_config import ConfigBase

from torch.multiprocessing import Lock


MODEL_DOWNLOAD_LOCK = Lock()


def run_model(model, inputs, layer):
  layers, _ = model(inputs)
  return layers[layer].cpu()


class EmbedderInterface:
  __REGISTRY = dict()

  class Config(ConfigBase):
    max_pieces: int = -1
    preproc_dir: str = "."
    use_cuda_if_available: bool = True
    embed_type: str = "BERTEmbed"

  """
  Naming convention:

    "encoding"  = text -> tokens -> ids, limit applied after encoding
    "decoding"  = text <- tokens <- ids, limit applied before decoding
    "embedding" = tokens -> vectors,     limit applied before embedding

  """
  def __init__(self, max_pieces: int = -1):
    self.pieces_slice = slice(max_pieces) if max_pieces > 0 else slice(None)

  @property
  def embed_dim(self):
    raise NotImplementedError()

  @property
  def n_vocab(self):
    raise NotImplementedError()

  @classmethod
  def register_class(cls):
    EmbedderInterface.__REGISTRY[cls.__name__] = cls

  @classmethod
  def from_config(cls, config: Config):
    return EmbedderInterface.__REGISTRY[config.embed_type].from_config(config)

  @abstractmethod
  def encode_text_as_ids(self, text: str) -> np.array:
    """
    Doesn't produce BOS, EOS ids.
    """
    raise NotImplementedError()

  @abstractmethod
  def encode_text_as_tokens(self, text: str) -> List[str]:
    """
    Doesn't produce BOS, EOS tokens.
    """
    raise NotImplementedError()

  @abstractmethod
  def tokenize(self, text: str) -> List[str]:
    """
    Alias for `encode_text_as_tokens`.
    Doesn't produce BOS, EOS tokens.
    """
    raise NotImplementedError()

  @abstractmethod
  def decode_ids_as_text(self, ids: List[int]) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS ids are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    raise NotImplementedError()

  @abstractmethod
  def decode_tokens_as_text(self, toks: List[str]) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS tokens are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    raise NotImplementedError()

  @abstractmethod
  def decode_id_as_token(self, id: int) -> str:
    raise NotImplementedError()

  @abstractmethod
  def decode_ids_as_tokens(self, ids: List[int], strip_special: bool = True) -> List[str]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """
    raise NotImplementedError()

  @abstractmethod
  def embed_tok(self, tok: str) -> np.array:
    """
    When given PAD, returns all zeros
    """
    raise NotImplementedError()

  @abstractmethod
  def embed_text(self, text: str) -> np.array:
    """
    Doesn't produce PAD, BOS, EOS embeddings.
    i.e. PAD, BOS, EOS are stripped out during tokenization before embedding.
    """
    raise NotImplementedError()

  @abstractmethod
  def embed_ids(self, ids: List[int], strip_special: bool = True) -> List[np.array]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """
    raise NotImplementedError()


class BERTEmbed(EmbedderInterface):
  """
  Naming convention:

    "encoding"  = text -> tokens -> ids, limit applied after encoding
    "decoding"  = text <- tokens <- ids, limit applied before decoding
    "embedding" = tokens -> vectors,     limit applied before embedding

  """

  class Config(EmbedderInterface.Config):
    pass

  def __init__(self, max_pieces: int = -1, use_cuda_if_available: bool = True):
    super().__init__(max_pieces=max_pieces)
    self.use_cuda_if_available = use_cuda_if_available
    self.tokenizer

  @property
  def tokenizer(self):
    """lazy model loading"""
    with MODEL_DOWNLOAD_LOCK:
      # use lock to ensure model isn't downloaded by two processes at once
      if not getattr(self, "_tokenizer", None):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.pad_token = '[PAD]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.unk_token = '[UNK]'
        self.pad_idx = self._tokenizer.vocab[self.pad_token]
        self.unk_idx = self._tokenizer.vocab[self.unk_token]

        # add EOS and BOS tokens to vocab by reusing unused slots
        self._tokenizer.basic_tokenizer.never_split += (self.eos_token, self.bos_token)
        vocab = self._tokenizer.vocab
        oldkey, newkey = '[unused1]', self.bos_token
        vocab = OrderedDict((newkey if k == oldkey else k, v) for k, v in vocab.items())
        oldkey, newkey = '[unused2]', self.eos_token
        vocab = OrderedDict((newkey if k == oldkey else k, v) for k, v in vocab.items())
        self._tokenizer.vocab = vocab
        self._tokenizer.wordpiece_tokenizer.vocab = vocab
        self.bos_idx = vocab[self.bos_token]
        self.eos_idx = vocab[self.eos_token]
        ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in vocab.items()])
        self._tokenizer.ids_to_tokens = ids_to_tokens
        self._tokenizer.wordpiece_tokenizer.ids_to_tokens = ids_to_tokens
    return self._tokenizer

  def __getstate__(self):
    return {k: v for k, v in self.__dict__.items() if k not in ("_model", "tokenizer")}

  def __setstate__(self, kwargs):
    self.__dict__.update(kwargs)

  @property
  def model(self):
    """lazy model loading"""
    with MODEL_DOWNLOAD_LOCK:
      # use lock to ensure model isn't downloaded by two processes at once
      if not getattr(self, "_model", None):
        self._model = BertModel.from_pretrained('bert-base-uncased')
        self._model.eval()
      assert self._model.config.hidden_size == self.embed_dim
    if cuda_utils.CUDA_ENABLED and self.use_cuda_if_available:
      self._model.cuda()
    return self._model

  @classmethod
  def from_config(cls, config: Config):
    return cls(config.max_pieces, use_cuda_if_available=config.use_cuda_if_available)

  @property
  def embed_dim(self):
    # we do not go through the model here, since we want to avoid loading it
    return 768

  @property
  def n_vocab(self):
    return len(self.tokenizer.vocab)

  def encode_text_as_ids(self, text: str) -> np.array:
    """
    Doesn't produce BOS, EOS ids.
    """
    ret = self.tokenizer.convert_tokens_to_ids(self.tokenize(text))[self.pieces_slice]
    return np.asarray(ret)

  def encode_text_as_tokens(self, text: str) -> List[str]:
    """
    Doesn't produce BOS, EOS tokens.
    """
    return self.tokenizer.tokenize(text)

  def tokenize(self, text: str) -> List[str]:
    """
    Alias for `encode_text_as_tokens`.
    Doesn't produce BOS, EOS tokens.
    """
    return self.tokenizer.tokenize(text)

  def decode_ids_as_text(self, ids: List[int], strip_special=True) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS ids are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    if strip_special:
      ids = [id for id in ids if id not in (self.pad_idx, self.bos_idx, self.eos_idx)]
    return self.decode_tokens_as_text(
      self.tokenizer.convert_ids_to_tokens(ids))

  def decode_tokens_as_text(self, toks: List[str]) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS tokens are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    if not toks:
      return ""
    ret = [toks[0]]
    for tok in toks[1:]:
      if tok.startswith('##'):
        ret.append(tok[2:])
      else:
        ret.extend((' ', tok))
    return "".join(ret)

  def decode_id_as_token(self, id: int) -> str:
    return self.tokenizer.convert_ids_to_tokens([id])[0]

  def decode_ids_as_tokens(self, ids: List[int], strip_special: bool = True) -> List[str]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """

    if strip_special:
      ids = [id for id in ids if id not in (self.pad_idx, self.bos_idx, self.eos_idx)]

    return [self.decode_id_as_token(ix) for ix in ids]

  def embed_tok(self, tok: str) -> np.array:
    """
    When given PAD, returns all zeros
    """
    raise RuntimeError("Embedding single tokens with BERT is likely not a good idea")

  def embed_text(self, text: str) -> np.array:
    """
    Doesn't produce PAD, BOS, EOS embeddings.
    i.e. PAD, BOS, EOS are stripped out during tokenization before embedding.
    """
    ids = self.encode_text_as_ids(text)
    return self.embed_ids(ids)

  def embed_ids_batch(self, ids: np.array) -> torch.tensor:
    """embeds a whole batch at once"""
    ids = torch.tensor(ids)
    with torch.no_grad():
      if cuda_utils.CUDA_ENABLED and self.use_cuda_if_available:
        ids = ids.cuda()
      return run_model(self.model, ids, -1).cpu()

  def embed_ids(self, ids: List[int], strip_special: bool = True) -> List[np.array]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """
    ids = [id for id in ids if id not in (self.pad_idx, self.eos_idx, self.bos_idx)]
    ids = torch.tensor(ids)
    with torch.no_grad():
      if cuda_utils.CUDA_ENABLED and self.use_cuda_if_available:
        ids = ids.cuda()
      return run_model(self.model, ids.unsqueeze(0), -1)[0].cpu()


class SentencepieceFasttextEmbed(EmbedderInterface):
  class Config(EmbedderInterface.Config):
    pass

  @classmethod
  def from_config(cls, config: Config):
    spm_model_file = os.path.join(config.preproc_dir, "spm.model")
    fasttext_model_file = os.path.join(config.preproc_dir, "fasttext-model.bin")
    return cls(spm_model_file, fasttext_model_file, config.max_pieces)

  def __init__(self, spm_model_file: str, fasttext_model_file: str = '', max_pieces: int = -1):
    super().__init__(max_pieces=max_pieces)

    self.spm = SentencePieceProcessor()
    self.spm.Load(spm_model_file)
    self.pad_idx = self.spm.pad_id()
    self.pad_token = self.spm.IdToPiece(self.pad_idx)
    self.unk_idx = self.spm.unk_id()
    self.unk_token = self.spm.IdToPiece(self.unk_idx)
    self.bos_idx = self.spm.bos_id()
    self.bos_token = self.spm.IdToPiece(self.bos_idx)
    self.eos_idx = self.spm.eos_id()
    self.eos_token = self.spm.IdToPiece(self.eos_idx)

    if fasttext_model_file:
      self.fasttext = fasttext.load_model(fasttext_model_file)

  @property
  def embed_dim(self):
    return self.fasttext.dim

  @property
  def n_vocab(self):
    return self.spm.get_piece_size()

  def encode_text_as_ids(self, text: str) -> np.array:
    """
    Doesn't produce BOS, EOS ids.
    """
    return np.asarray(self.spm.EncodeAsIds(text)[self.pieces_slice], dtype=np.int32)

  def encode_text_as_tokens(self, text: str) -> List[str]:
    """
    Doesn't produce BOS, EOS tokens.
    """
    return self.spm.EncodeAsPieces(text)[self.pieces_slice]

  def tokenize(self, text: str) -> List[str]:
    """
    Alias for `encode_text_as_tokens`.
    Doesn't produce BOS, EOS tokens.
    """
    return self.encode_text_as_tokens(text)[self.pieces_slice]

  def decode_ids_as_text(self, ids: List[int], strip_special=True) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS ids are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    if strip_special:
      ids = [int(id) for id in ids if id not in (self.pad_idx, self.bos_idx, self.eos_idx)]
    else:
      ids = [int(id) for id in ids]
    return self.spm.DecodeIds(ids)

  def decode_tokens_as_text(self, toks: List[str]) -> str:
    """
    Doesn't produce PAD, BOS, or EOS text.
    i.e. PAD, BOS, EOS tokens are stripped out before decoding.
    UNK is decoded but unintelligible.
    """
    return self.spm.DecodePieces(toks[self.pieces_slice])

  @functools.lru_cache(maxsize=1024)
  def decode_id_as_token(self, id: int) -> str:
    return self.spm.IdToPiece(id)

  def decode_ids_as_tokens(self, ids: List[int], strip_special: bool = True) -> List[str]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """
    if strip_special:
      ids = [id for id in ids if id not in (self.pad_idx, self.bos_idx, self.eos_idx)]
    return [self.decode_id_as_token(int(ix)) for ix in ids]

  @functools.lru_cache(maxsize=1024)
  def embed_tok(self, tok: str) -> np.array:
    """
    When given PAD, returns all zeros
    """
    if tok == self.pad_token:
      return np.zeros(self.fasttext.dim)
    return np.asarray(self.fasttext[tok])

  def embed_text(self, text: str) -> np.array:
    """
    Doesn't produce PAD, BOS, EOS embeddings.
    i.e. PAD, BOS, EOS are stripped out during tokenization before embedding.
    """
    return np.asarray([self.embed_tok(tok) for tok in self.tokenize(text)])

  def embed_ids(self, ids: List[int], strip_special: bool = True) -> List[np.array]:
    """
    By default, doesn't produce PAD, BOS, EOS tokens.

    Avoids problematic intermediate string representation that causes length mismatch.
    In other words, SentencePiece isn't isomorphic with respect to the string representation.
    """
    return [self.embed_tok(t) for t in self.decode_ids_as_tokens(ids, strip_special=strip_special)]

  def embed_ids_batch(self, ids: np.array) -> torch.tensor:
    emb = [self.embed_ids(turn, strip_special=False) for turn in ids]
    emb = torch.tensor(emb)
    return emb


SentencepieceFasttextEmbed.register_class()
BERTEmbed.register_class()
