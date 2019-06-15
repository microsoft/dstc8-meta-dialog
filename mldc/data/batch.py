from itertools import chain
import numpy as np


def rec_max_shape(data):
  """
  given a list of list of lists...., return the maximum length at each level.
  """
  res = [len(data)]
  while hasattr(data[0], '__iter__'):
    res.append(max(len(x) for x in data))
    data = [x for x in chain(*data)]
  return res


def left_pad_sequences(sequences, max_len=None, value=0, eos=False, dtype=None):
  if max_len is None:
    max_len = max(len(s) for s in sequences)
  try:
    dummy_shape = np.array(sequences[0][0]).shape
  except:  # noqa: E722
    dummy_shape = []
  shape = [s for s in chain([len(sequences), max_len], dummy_shape)]
  batch = value * np.ones(shape, dtype=dtype)
  for i, seq in enumerate(sequences):
    batch[i, max_len - len(seq):] = seq
  return batch


def right_pad_sequences(sequences, max_len=None, value=0, eos=False, dtype=None):
  if max_len is None:
    max_len = max(len(s) for s in sequences)
  try:
    dummy_shape = np.array(sequences[0][0]).shape
  except:  # noqa: E722
    dummy_shape = []
  shape = [s for s in chain([len(sequences), max_len], dummy_shape)]
  batch = value * np.ones(shape, dtype=dtype)
  # if eos:  # not needed when we use masking
  #     batch[:,0,:] = 1
  for i, seq in enumerate(sequences):
    batch[i, :len(seq)] = seq
  return batch


def pad_recursive(data, default=None, dir='right', value=0, dtype=None):
  """
  pad a list of lists of lists... of scalars to a dense numpy array
  """
  shape = rec_max_shape(data)
  dirs = dir
  if isinstance(dir, str):
    dirs = [dir]
  dirs = dirs + [dirs[-1]] * (len(shape) - len(dirs))
  if default is None:
    default = []
  for i, v in enumerate(default):
    if v is not None:
      shape[i] = v

  def f(data, shape, dir, default):
    if len(shape) == 0:
      return data
    data = [f(d, shape[1:], dir[1:], default) for d in data]
    if dir[0] == 'right':
      data = right_pad_sequences(data, shape[0], value=value, dtype=dtype)
    else:
      data = left_pad_sequences(data, shape[0], value=value, dtype=dtype)
    return data

  return np.asarray(f(data, shape[1:], dirs[1:], default), dtype=dtype)


def get_max_size(a, dim):
  if dim == 1:
    return len(a)
  return max([get_max_size(x, dim - 1) for x in a])


def right_pad_fixed_shape(data, max_shape, dtype, value=0):

  def drill_pad(data, max_shape, dest_view):
    # Pop a dim off the shape corresponding to the dim we'll iterate over
    max_shape = max_shape[1:]
    for i, arr in enumerate(data):
      # assign the lowest level array-like object to a 'row' of the view
      if len(max_shape) == 1:
        dest_view[i, :len(arr)] = arr
      else:
        sub_view = dest_view[i, ...]
        drill_pad(arr, max_shape, sub_view)

  batchtensor = np.full(max_shape, value, dtype=dtype)
  drill_pad(data, max_shape, batchtensor)
  return batchtensor
