import torch
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_
# from torch2trt_dynamic.torch2trt_dynamic.plugins import create_batchednms_plugin
# from torch2trt_dynamic.torch2trt_dynamic.plugins import create_batch_nms_plugin
from ..plugins import create_batch_nms_plugin


@tensorrt_converter('mmdet2trt.core.post_processing.batched_nms.BatchedNMS.forward')
def convert_batchednms(ctx):
    module = ctx.method_args[0]
    bboxes = ctx.method_args[1]
    scores = ctx.method_args[2]

    iou_thresh = module.iou_thresh
    score_thresh = module.score_thresh
    max_output_boxes_per_class = module.max_output_boxes_per_class

    bboxes_trt = trt_(ctx.network, bboxes)
    scores_trt = trt_(ctx.network, scores)
    output = ctx.method_return
    
    plugin = create_batch_nms_plugin(
        'batchednms_' + str(id(module)),
        iou_thresh,
        score_thresh,
        max_output_boxes_per_class)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[bboxes_trt, scores_trt], plugin=plugin)

    if isinstance(output, torch.Tensor):
        output._trt = custom_layer.get_output(0)
    else:
        # for i in range(len(output)):
        #     output[i]._trt = custom_layer.get_output(i)
        raise ValueError
