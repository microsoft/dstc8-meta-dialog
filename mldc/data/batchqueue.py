import logging
import pickle

import time
from threading import Thread
from multiprocessing import Queue
import torch.multiprocessing
from queue import Full
from iterable_queue import IterableQueue
from pytext.workflow import _set_cuda


mp = torch.multiprocessing.get_context('spawn')
LOG = logging.getLogger("mldc.batchq")


class BatchProcessor:

  def __init__(self):
    pass

  def process_batch(self, batch):
    pass


class BatchQueue:
  """
  Wraps an iterator with a parallel asynchronous mechanism for queuing multiple batches at the same time.
  Implemented as a pool of resusable processes over two iterable threadsafe queues, avoiding process creation
  and setup (e.g. load fasttext) overhead.

  The producer process takes a batch from the base iter puts it on `todoq` (producer).
  A worker process takes a batch off `todoq` (consumer).
  The worker process processes the batch and places the result on `doneq` (producer).
  The main process takes a processed batch off `doneq` (consumer).

  To cleanly end iteration prematurely, call close() on the BatchQueue object.
  """

  @staticmethod
  def _worker_loop(todoq, doneq, wid, proc_class, *args, **kwargs):
    device_id = 0
    world_size = 1
    _set_cuda(True, device_id, world_size)

    # setup
    processor = proc_class(*args, **kwargs)

    for raw_batch in todoq:
      processed_batch = processor.process_batch(raw_batch)
      doneq.put(pickle.dumps(processed_batch))

    doneq.close()

  def __init__(self, base_iterator, n_batches, batch_processor_cls, enqueue_fn=None,
               n_workers=3, qcap=5, *args, **kwargs):
    LOG.info("BatchQueue: n_workers=%d max queue size=%d", n_workers, qcap)
    self._base_iterator = base_iterator
    self._n_batches = n_batches
    self._todoq = IterableQueue(qcap)
    self._doneq = IterableQueue(qcap)
    self._enqueue_fn = enqueue_fn
    self._workers = []
    self._end_sigq = Queue()

    # use threading here to avoid pickling, particularly since this process is fairly lightweight.
    self._producer = Thread(target=self.enq_examples_for_workers, args=(self._todoq.get_producer(), self._end_sigq,))

    for wid in range(1, n_workers + 1):
      worker_todo = self._todoq.get_consumer()
      worker_done = self._doneq.get_producer()
      w = mp.Process(target=BatchQueue._worker_loop,
                     args=(worker_todo, worker_done, wid, batch_processor_cls, *args),
                     kwargs=kwargs)
      w.start()
      self._workers.append(w)

    self._main_done = self._doneq.get_consumer()
    self._producer.start()

    self._todoq.close()
    self._doneq.close()

  def enq_examples_for_workers(self, todo_queue, end_queue):
    for bid, batch in enumerate(self._base_iterator):
      if self._enqueue_fn:
        batch = self._enqueue_fn(batch)
      while True:
        try:
          todo_queue.put(batch, block=True, timeout=1)
          break
        except Full:
          # try again, but before that check whether stop was requested
          time.sleep(0)  # yield control to other threads for now
          pass
        finally:
          # stop putting stuff in the queue if end signaled
          if not end_queue.empty():
            todo_queue.close()
            return
    todo_queue.close()

  def close(self):
    """ Note: must be called explicitly since putting this in `__del__` doesn't work."""
    # stop generating data
    self._end_sigq.put("stop")

    # Drain the queue
    for _ in self._main_done:
      pass

    # note this cannot be done before draining
    self._producer.join()

  def __iter__(self):
    for item in self._main_done:
      yield pickle.loads(item)

  def __len__(self):
    return self._n_batches
