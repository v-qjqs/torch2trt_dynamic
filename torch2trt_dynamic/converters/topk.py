import tensorrt as trt
import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter, trt_, torch_dim_to_trt_axes)

@tensorrt_converter('torch.topk')
@tensorrt_converter('torch.Tensor.topk')
def convert_topk(ctx):
  input = ctx.method_args[0]
  output = ctx.method_return
  k = get_arg(ctx, 'k', 1, None)
  axis = get_arg(ctx, 'dim', 2, default=len(input.shape) - 1)
  largest = get_arg(ctx, 'largest', 3, True)

  if axis < 0:
    axis = len(input.shape) + axis
  assert k is not None
  if k > 1024:
    print('warning: topk = ' + k + ' > 1024 is not allowed in TensorRT.')
    # k = 3840
    raise ValueError
  
  topkOp = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN
  input_trt = trt_(ctx.network, input)

  # NOTE TODO check why 1<<axis instead of axis is needed here
  layer = ctx.network.add_topk(input_trt, topkOp, k, 1 << axis)
  output_value_trt = layer.get_output(0)
  output_index_trt = layer.get_output(1)
  output[0]._trt = output_value_trt
  output[1]._trt = output_index_trt
