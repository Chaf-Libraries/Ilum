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
	PipelineState(RHIDevice *device, Descriptor *descriptor);

	virtual ~PipelineState() override;

	VkPipelineLayout GetPipelineLayout() const;

	VkPipeline GetPipeline() const;

  private:
	VkPipelineLayout CreatePipelineLayout() const;
	VkPipeline       CreatePipeline() const;

  private:
	const ShaderMeta m_meta;

	Descriptor *m_descriptor = nullptr;
};
}        // namespace Ilum::Vulkan