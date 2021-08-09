import torch
import tensorrt as trt
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter, trt_, torch_dim_to_trt_axes)
from torch2trt_dynamic.module_test import add_module_test
from .topk import convert_topk
from .squeeze import convert_squeeze
# from .unary import UnaryModule


def __convert_max_elementwise(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = trt_(ctx.network, input_a, input_b)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.MAX)
    output._trt = layer.get_output(0)
    

def __convert_max_reduce(ctx):
    input = ctx.method_args[0]
    output = ctx.method_return
    keepdim = get_arg(ctx, 'keepdim', pos=2, default=False)
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    if isinstance(output, torch.Tensor):
        dim = tuple(range(0, len(input.shape)))
        input_trt = trt_(ctx.network, input)
        # TODO check torch_dim_to_trt_axes
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.MAX, torch_dim_to_trt_axes(dim), keepdim)
        output._trt = layer.get_output(0)
        return

    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs

    # topk
    assert dim is not None
    topk_output = input.topk(1, dim)
    topk_input = [input, 1, dim]
    ctx.method_args = topk_input
    ctx.method_kwargs = {}
    ctx.method_return = topk_output
    convert_topk(ctx)
    topk_value = ctx.method_return[0]
    topk_index = ctx.method_return[1]

    # keepdim
    if not keepdim:
        topk_index_squeeze = topk_index.squeeze(dim)
        ctx.method_args = [topk_index, dim]
        ctx.method_return = topk_index_squeeze
        convert_squeeze(ctx)

        topk_value_squeeze = topk_value.squeeze(dim)
        ctx.method_args = [topk_value, dim]
        ctx.method_return = topk_value_squeeze
        convert_squeeze(ctx)

        topk_index = topk_index_squeeze
        topk_value = topk_value_squeeze

    output[0]._trt  = topk_value._trt
    output[1]._trt = topk_index._trt
    ctx.method_return = output

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs


@tensorrt_converter('torch.max')
@tensorrt_converter('torch.Tensor.max')
def convert_max(ctx):
    if len(ctx.method_args) > 1 and isinstance(ctx.method_args[1], torch.Tensor):
        __convert_max_elementwise(ctx)
    else:
        __convert_max_reduce(ctx)
        

# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
# def test_max_reduce_dim1():
#     return UnaryModule(lambda x: torch.max(x, 1)[0])


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
# def test_max_reduce_dim22():
#     return UnaryModule(lambda x: torch.max(x, 2)[0])


# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
# def test_max_reduce_dim1_keepdim():
#     return UnaryModule(lambda x: torch.max(x, 1, keepdim=True)[0])


# class MaxElementwise(torch.nn.Module):
#     def forward(self, x, y):
#         return torch.max(x, y)
    
    
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1, 3, 3)])
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3), (1,)]) # broadcast
# @add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3), (1, 3, 3)]) # broadcast
# def test_max_elementwise():
#     return MaxElementwise()
