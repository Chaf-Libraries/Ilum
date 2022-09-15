#pragma once

#include "RHI/RHIPipelineState.hpp"
#include "RHI/RHIShader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Descriptor;
class RenderTarget;

struct ShaderBindingTable
{
	const VkStridedDeviceAddressRegionKHR *raygen   = nullptr;
	const VkStridedDeviceAddressRegionKHR *miss     = nullptr;
	const VkStridedDeviceAddressRegionKHR *hit      = nullptr;
	const VkStridedDeviceAddressRegionKHR *callable = nullptr;
};

class PipelineState : public RHIPipelineState
{
  public:
	PipelineState(RHIDevice *device);

	virtual ~PipelineState() override;

	VkPipelineLayout GetPipelineLayout(Descriptor *descriptor);

	VkPipeline GetPipeline(Descriptor *descriptor, RenderTarget *render_target);

	ShaderBindingTable GetShaderBindingTable(VkPipeline pipeline);

	VkPipelineBindPoint GetPipelineBindPoint() const;

  private:
	VkPipelineLayout CreatePipelineLayout(Descriptor *descriptor);
	VkPipeline       CreateGraphicsPipeline(Descriptor *descriptor, RenderTarget *render_target);
	VkPipeline       CreateComputePipeline(Descriptor *descriptor);
	VkPipeline       CreateRayTracingPipeline(Descriptor *descriptor);
};
}        // namespace Ilum::Vulkan