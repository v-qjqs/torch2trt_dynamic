# include "torch_gather_plugin.hpp"
# include "nms_plugin.hpp"
# include "trt_plugins.hpp"

namespace amirstan {
namespace plugin {
// REGISTER_TENSORRT_PLUGIN(AdaptivePoolPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(BatchedNMSPluginCustomCreator);
// REGISTER_TENSORRT_PLUGIN(CarafeFeatureReassemblePluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(DeformablePoolPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(Delta2BBoxPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(GridAnchorDynamicPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(GridSamplePluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(GroupNormPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(MeshGridPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(RoiExtractorPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(RoiPoolPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(TorchBmmPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(TorchCumMaxMinPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(TorchCumPluginDynamicCreator);
// REGISTER_TENSORRT_PLUGIN(TorchEmbeddingPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchGatherPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(TorchNMSPluginDynamicCreator);
}  // namespace plugin
}  // namespace amirstan

extern "C" {

bool initLibInferPlugins() { return true; }
}  // extern "C"
