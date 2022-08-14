#pragma once

#include "RHI/RHIPipelineState.hpp"
#include "RHI/RHIShader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Descriptor;

class PipelineState : public RHIPipelineState
{
  public:
	PipelineState(RHIDevice *device, Descriptor* descriptor);

	virtual ~PipelineState() override;

	VkPipelineLayout GetPipelineLayout() const;

	VkPipeline GetPipeline() const;

  private:
	VkPipeline       m_pipeline        = VK_NULL_HANDLE;
	VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan