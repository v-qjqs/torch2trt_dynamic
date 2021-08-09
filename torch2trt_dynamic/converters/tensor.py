import numpy as np
from numpy.lib import isin
import tensorrt as trt
import torch
# from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import get_arg, tensorrt_converter, trt_, torch_dtype_to_trt
from torch2trt_dynamic.converters.size import IntWarper


@tensorrt_converter('torch.tensor')
def convert_tensor(ctx):
  input = ctx.method_args[0]
  output = ctx.method_return
  torch_dtype = output.dtype
  if torch_dtype in [torch.long, torch.int64]:
    torch_dtype = torch.int
  elif torch_dtype in [torch.float64]:
    torch_dtype = torch.float
  trt_dtype = torch_dtype_to_trt(torch_dtype)

  is_const = True
  if isinstance(input, (tuple, list)):
    for input_i in input:
      if isinstance(input_i, IntWarper):
        is_const = False
        break
  else:
    if hasattr(input, '_trt'):
      is_const = False
  if is_const:
    return
  #   weight = output.detach().cpu().numpy()
  #   if weight.dtype == np.float64:
  #       weight = weight.astype(np.float32)
  #   elif weight.dtype == np.int64:
  #       weight = weight.astype(np.int32)
  #   layer = ctx.network.add_constant(output.shape, weight)
  else:
    if isinstance(input, (tuple, list)):
      input_trt = []
      trt_inputs = [trt_(ctx.network, input_i) for input_i in input]
      layer = ctx.network.add_concatenation(inputs=trt_inputs)
      layer.axis = 0

      # # convert 2 / to prevent warning, might remove in future version
      # val_trt = layer.get_output(0)
      # layer = ctx.network.add_elementwise(
      #     val_trt, trt_(ctx.network, torch.zeros((1, ), dtype=torch.int32)),
      #     trt.ElementWiseOperation.SUM)
      # layer.set_output_type(0, trt_dtype)

    else:
      input_trt = trt_(ctx.network, input)
      layer = ctx.network.add_identity(input_trt)

  if output.ndim == 0:
    # scale tensor
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple()
  layer.set_output_type(0, trt_dtype)
  output_trt = layer.get_output(0)
  output._trt = output_trt


# @tensorrt_converter('torch.tensor')
# def convert_tensor(ctx):
#   input = ctx.method_args[0]
#   output = ctx.method_return
#   torch_dtype = output.dtype
#   if torch_dtype in [torch.long, torch.int64]:
#     torch_dtype = torch.int
#   elif torch_dtype in [torch.float64]:
#     torch_dtype = torch.float
#   trt_dtype = torch_dtype_to_trt(torch_dtype)

#   is_const = True
#   if hasattr(input, '_trt'):
#     is_const = False
#   if is_const:
#     return
#   #   weight = output.detach().cpu().numpy()
#   #   if weight.dtype == np.float64:
#   #       weight = weight.astype(np.float32)
#   #   elif weight.dtype == np.int64:
#   #       weight = weight.astype(np.int32)
#   #   layer = ctx.network.add_constant(output.shape, weight)
#   else:
#     input_trt = trt_(ctx.network, input)
#     layer = ctx.network.add_identity(input_trt)
#   if output.ndim == 0:
#     # scale tensor
#     layer = ctx.network.add_shuffle(layer.get_output(0))
#     layer.reshape_dims = tuple()
#   layer.set_output_type(0, trt_dtype)
#   output_trt = layer.get_output(0)
#   output._trt = output_trt
