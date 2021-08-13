// Copyright (c) OpenMMLab. All rights reserved
#include "trt_plugin.hpp"

#include "trt_batch_nms.hpp"

REGISTER_TENSORRT_PLUGIN(NonMaxSuppressionDynamicCreator);

extern "C" {
bool initLibInferPlugins() { return true; }
}  // extern "C"
