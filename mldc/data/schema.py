""""
defines a class that maps to the JSON input format and can be used with pydantic.
"""
import json
import os
import pickle
from hashlib import md5
from typing import List, Optional
from pydantic import BaseModel

from mldc.util import NLGEvalOutput


class MetaDlgDataDialog(BaseModel):
  id: Optional[str]
  domain: str = ""
  task_id: str = ""
  user_id: str = ""
  bot_id: str = ""
  turns: List[str]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


class MetaDlgDataDialogList(BaseModel):
  dialogs: List[MetaDlgDataDialog]


class PartitionSpec(BaseModel):
  domains: List[str] = []
  tasks: List[str] = []
  paths: List[str] = []

  def _asdict(self):
    # convert to list for json-serializability
    return dict(domains=self.domains, tasks=self.tasks, paths=self.paths)

  # the next few fields/functions are here to make PartitionSpec behave like
  # a pytext ConfigBase object. This way, we can use it directly in a task
  # config. It would be easier if we could just inherit from ConfigBase,
  # but alas, ConfigBase's metaclass is not a metaclass of BaseModel.
  _field_types = __annotations__  # noqa

  @property
  def _fields(cls):
    return cls.__annotations__.keys()

  @property
  def _field_defaults(cls):
    _, defaults = cls.annotations_and_defaults()
    return defaults

  def is_ok(self, dlg: MetaDlgDataDialog):
    if self.tasks and dlg.task_id not in self.tasks:
      return False
    if self.domains and dlg.domain not in self.domains:
      return False
    return True

  def __bool__(self):
    return True if self.domains or self.tasks or self.paths else False

  def add(self, other):
    self.domains = list(set(self.domains + other.domains))
    self.tasks = list(set(self.tasks + other.tasks))
    self.paths = list(set(self.paths + other.paths))

  @classmethod
  def from_paths(cls, paths):
    return cls(domains=[], paths=paths, tasks=[])

  def iterate_paths(self):
    for path in self.paths:
      yield path, PartitionSpec(domains=[NLGEvalOutput._domain_name(path)],
                                paths=[path],
                                tasks=self.tasks)

  def checksum(self, zipfile, featurizer_config, text_embedder_cfg):
    checksum = md5(json.dumps(featurizer_config._asdict(), sort_keys=True).encode('utf-8'))
    text_embedder_cfg = text_embedder_cfg._asdict()
    del text_embedder_cfg['preproc_dir']
    del text_embedder_cfg['use_cuda_if_available']
    checksum.update(json.dumps(text_embedder_cfg, sort_keys=True).encode('utf-8'))
    md5file = zipfile + ".md5"
    # if md5file exists and is newer than zipfile, read md5 sum from it
    # else calculate it for the zipfile.
    if os.path.exists(md5file) and os.path.getmtime(zipfile) <= os.path.getmtime(md5file):
      with open(md5file, 'rt') as f:
        checksum.update(f.read().split()[0].strip().encode('utf-8'))
    else:
      with open(zipfile, 'rb') as f:
        checksum.update(md5(f.read()).hexdigest().encode('utf-8'))
    checksum.update(pickle.dumps(sorted(self.domains)))
    checksum.update(pickle.dumps(sorted(self.paths)))
    checksum.update(pickle.dumps(sorted(self.tasks)))
    return checksum.hexdigest()


class DataSpec(BaseModel):
  train: PartitionSpec = PartitionSpec()
  validation: PartitionSpec = PartitionSpec()
  test: PartitionSpec = PartitionSpec()

  def unpack_domains(self):
    return [list(p) for p in (self.train.domains, self.validation.domains, self.test.domains)]

  def unpack_tasks(self):
    return [list(p) for p in (self.train.tasks, self.validation.tasks, self.test.tasks)]

  def unpack_paths(self):
    return [list(p) for p in (self.train.paths, self.validation.paths, self.test.paths)]

  def unpack(self):
    return self.train._asdict(), self.validation._asdict(), self.test._asdict()

  @classmethod
  def load(cls, f):
    kwargs = json.load(f)
    # This just works with Pydantic
    return cls(**kwargs)

  def add(self, other):
    self.train.add(other.train)
    self.validation.add(other.validation)
    self.test.add(other.test)
  