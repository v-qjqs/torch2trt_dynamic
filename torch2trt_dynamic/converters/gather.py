# from ..plugins import create_torchgather_plugin
# from ..torch2trt_dynamic import get_arg, tensorrt_converter, trt_


# @tensorrt_converter('torch.Tensor.gather')
# @tensorrt_converter('torch.gather')
# def convert_gather(ctx):
#     inputs = ctx.method_args[0]
#     dim = get_arg(ctx, 'dim', pos=1, default=0)
#     index = get_arg(ctx, 'index', pos=2, default=None)
#     output = ctx.method_return

#     inputs_trt = trt_(ctx.network, inputs)
#     index_trt = trt_(ctx.network, index)

#     plugin = create_torchgather_plugin(
#         'torch_gather_' + str(id(inputs)), dim=dim)

#     layer = ctx.network.add_plugin_v2(
#         inputs=[inputs_trt, index_trt], plugin=plugin)

#     output._trt = layer.get_output(0)


# # @tensorrt_converter('torch.gather')
# # def convert_gather(ctx):
# #   input = ctx.method_args
# #   output = ctx.method_return
# #   axis = get_arg(ctx, 'dim', 1, None)
# #   index = get_arg(ctx, 'index', 2, None)
# #   assert axis is not None and index is not None
# #   if axis < 0:
# #     axis = len(input.shape) + axis

# #   input_trt = trt_(ctx.network, input)
# #   index_trt = trt_(ctx.network, index)
# #   layer = ctx.network.add_gather(input_trt, index_trt, axis)
# #   output._trt = layer.get_output(0)
