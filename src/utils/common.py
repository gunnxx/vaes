from typing import Any, Dict, List, Tuple

import torch.nn as nn
import src.model.residual_layer as rl

## just an alias
model_args_dtype = List[Tuple[str, Dict[str, Any]]]

"""
"""
def instantiate_layer(ltype: str, lparams: Dict[str, Any]) -> nn.Module:
  ltype = ltype.lower()

  ## main layer
  if ltype == "linear":
    layer = nn.Linear(**lparams)
  elif ltype == "conv2d":
    layer = nn.Conv2d(**lparams)
  elif ltype == "convtranspose2d":
    layer = nn.ConvTranspose2d(**lparams)
  elif ltype == "upsample":
    layer = nn.Upsample(**lparams)
  elif ltype == "batchnorm2d":
    layer = nn.BatchNorm2d(**lparams)
  elif ltype == "batchnorm1d":
    layer = nn.BatchNorm1d(**lparams)
  elif ltype == "flatten":
    layer = nn.Flatten()
  elif ltype == "residuallayer":
    layer = rl.ResidualLayer(**lparams)
  
  ## activation function
  elif ltype == "activation":
    if lparams == "relu":
      layer = nn.ReLU()
    elif lparams == "sigmoid":
      layer = nn.Sigmoid()
    elif lparams == "tanh":
      layer = nn.Tanh()
    elif lparams == "leakyrelu":
      layer = nn.LeakyReLU(0.1)
  
  ## unregistered layer
  else:
    raise KeyError("Layer type is not recognized.")
  
  return layer

"""
"""
def dcgan_weights_init(m: nn.Module) -> None:
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    nn.init.normal_(m.weight.data, 0., .02)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.normal_(m.weight.data, 1., .02)
    nn.init.normal_(m.bias.data)