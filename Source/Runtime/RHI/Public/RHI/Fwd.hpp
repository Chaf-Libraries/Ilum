#pragma once

#include "RHIDefinitions.hpp"

#include <Core/Core.hpp>
#include <Core/Hash.hpp>
#include <Core/Window.hpp>

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

namespace Ilum
{
class RHIAccelerationStructure;
class RHIBuffer;
class RHICommand;
class RHIDescriptor;
class RHIDevice;
class RHIFrame;
class RHIPipelineState;
class RHIProfiler;
class RHIQueue;
class RHIRenderTarget;
class RHISampler;
class RHIShader;
class RHISwapchain;
class RHIFence;
class RHISemaphore;
class RHITexture;
struct ShaderMeta;
}        // namespace Ilum