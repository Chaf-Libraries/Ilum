#include "PipelineState.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
static std::unordered_map<size_t, VkPipeline> PipelineCache;

}