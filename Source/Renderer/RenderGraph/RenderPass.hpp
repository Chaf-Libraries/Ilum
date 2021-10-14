#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Descriptor/DescriptorBinding.hpp"

namespace Ilum
{
class PipelineState;
class RenderGraph;

struct PassNative
{
	VkRenderPass               render_pass = VK_NULL_HANDLE;
	std::vector<DescriptorSet> descriptor_sets;
	VkFramebuffer              frame_buffer    = VK_NULL_HANDLE;
	VkPipeline                 pipeline        = VK_NULL_HANDLE;
	VkPipelineLayout           pipeline_layout = VK_NULL_HANDLE;
	VkPipelineBindPoint        bind_point;
	VkRect2D                   render_area;
	std::vector<VkClearValue>  clear_values;
};

struct RenderPassState
{
	RenderGraph &        graph;
	const CommandBuffer &command_buffer;
	const PassNative &   pass;

	const Image &getAttachment(const std::string &name);
};

using ResolveState = ResolveInfo;

class RenderPass
{
  public:
	virtual ~RenderPass() = default;

	virtual void setupPipeline(PipelineState &state){};

	virtual void resolveResources(ResolveState &resolve){};

	virtual void initialize(RenderPassState &state){};

	virtual void render(RenderPassState &state){};
};
}        // namespace Ilum