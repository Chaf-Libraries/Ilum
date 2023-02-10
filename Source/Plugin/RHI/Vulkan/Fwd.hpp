#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <volk.h>

#include "vk_mem_alloc.h"

#include <Core/Core.hpp>

#include <RHI/RHIAccelerationStructure.hpp>
#include <RHI/RHIBuffer.hpp>
#include <RHI/RHICommand.hpp>
#include <RHI/RHIDefinitions.hpp>
#include <RHI/RHIDescriptor.hpp>
#include <RHI/RHIDevice.hpp>
#include <RHI/RHIFrame.hpp>
#include <RHI/RHIPipelineState.hpp>
#include <RHI/RHIProfiler.hpp>
#include <RHI/RHIQueue.hpp>
#include <RHI/RHIRenderTarget.hpp>
#include <RHI/RHISampler.hpp>
#include <RHI/RHIShader.hpp>
#include <RHI/RHISwapchain.hpp>
#include <RHI/RHISynchronization.hpp>
#include <RHI/RHITexture.hpp>

#include "Definitions.hpp"

#ifdef _WIN64
#	include <Windows.h>
#endif        // _WIN64

namespace Ilum
{
namespace Vulkan
{
class AccelerationStructure;
class Buffer;
class Command;
class Descriptor;
class Device;
class Frame;
class PipelineState;
class Profiler;
class Queue;
class RenderTarget;
class Sampler;
class Shader;
class Swapchain;
class Fence;
class Semaphore;
class Texture;
}        // namespace Vulkan
}        // namespace Ilum