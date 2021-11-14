from typing import List

class EarlyStop:
  """
  """
  def __init__(self, patience: int = 3, warmup: int = 10) -> None:
    self.patience = patience
    self.warmup = warmup
    
    self.iteration = 0
    self.metrics = []

  """
  """
  def __call__(self, metric: float) -> None:
    self.metrics.append(metric)
    if len(self.metrics) > (self.patience + 1):
      self.metrics.pop(0)
    
    self.iteration += 1
  
  """
  """
  def is_stop(self) -> bool:
    ## return False in warmup phase
    if self.iteration < self.warmup:
      return False
    
    ## check progress among the last `self.patience`
    for metric in self.metrics[1:]:
      if metric < self.metrics[0]:
        return False
    
    ## no progress
    return True