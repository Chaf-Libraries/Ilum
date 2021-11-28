#include "CopyPass.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

namespace Ilum::pass
{
CopyPass::CopyPass()
{
	
}

void CopyPass::setupPipeline(PipelineState &state)
{


	state.addDependency("gbuffer - depth", VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	state.addDependency("depth - buffer", VK_IMAGE_USAGE_SAMPLED_BIT);
}

void CopyPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("depth - buffer", *Renderer::instance()->Last_Frame.depth_buffer);
}

void CopyPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	cmd_buffer.copyImage(
	    ImageInfo{state.graph.getAttachment("gbuffer - depth"), VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
	    ImageInfo{*Renderer::instance()->Last_Frame.depth_buffer, VK_IMAGE_USAGE_SAMPLED_BIT});

	cmd_buffer.transferLayout(*Renderer::instance()->Last_Frame.depth_buffer, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_SAMPLED_BIT);
}
}        // namespace Ilum::pass