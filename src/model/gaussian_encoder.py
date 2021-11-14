import torch
import torch.distributions as d
import torch.nn as nn

from src.utils.common import model_args_dtype, instantiate_layer

class GaussianEncoder(nn.Module):
  """
  """
  def __init__(self, model_args: model_args_dtype) -> None:
    super(GaussianEncoder, self).__init__()

    self.base_layers = nn.Sequential([instantiate_layer(lt, lp) for lt, lp in model_args[:-1]])
    self.mu_layer = nn.Linear(**model_args[-1][1])
    self.logvar_layer = nn.Linear(**model_args[-1][1])
  
  """
  """
  def forward(self, x: torch.Tensor) -> d.Normal:
    ## base network
    h = self.base_layers(x)
    
    ## mu and std of gaussian
    mu = self.mu_layer(h)
    logvar = self.logvar_layer(h)
    std = torch.exp(logvar * 0.5)

    ## note that `mu.shape = (batch_dim, embed_dim)`
    ## it adjustst the appropriate `batch_shape` and `event_shape`
    return d.Independent(d.Normal(mu, std), 1)