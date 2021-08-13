import numpy as np
import tensorrt as trt


def create_batch_nms_plugin(layer_name, iou_threshold, score_threshold, max_output_boxes_per_class):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'NonMaxSuppression', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_iou_threshold = trt.PluginField(
        'iou_threshold', np.array([iou_threshold], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_iou_threshold)

    pf_score_threshold = trt.PluginField(
        'score_threshold', np.array([score_threshold], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_score_threshold)

    pf_max_output_boxes_per_class = trt.PluginField(
        'max_output_boxes_per_class', np.array([max_output_boxes_per_class], dtype=np.float32),
        trt.PluginFieldType.FLOAT32)
    pfc.append(pf_max_output_boxes_per_class)
    # center_point_box, offset
    # boxes, scores,

    return creator.create_plugin(layer_name, pfc)
