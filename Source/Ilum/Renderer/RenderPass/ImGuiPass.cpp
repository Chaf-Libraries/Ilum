#include "ImGuiPass.hpp"

#include <Graphics/Device/Instance.hpp>
#include <Graphics/Device/Device.hpp>
#include <Graphics/Device/PhysicalDevice.hpp>
#include <Graphics/Device/Surface.hpp>
#include <Graphics/RenderContext.hpp>

#include "Device/Swapchain.hpp"

#include "Graphics/Descriptor/DescriptorCache.hpp"
#include "Graphics/Descriptor/DescriptorPool.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/PipelineState.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/RenderGraph/RenderGraphBuilder.hpp"
#include "Renderer/Renderer.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>

namespace Ilum::pass
{
ImGuiPass::ImGuiPass(const std::string &output_name, const std::string &view_name, AttachmentState state) :
    m_output(output_name), m_view(view_name), m_attachment_state(state)
{
}

void ImGuiPass::setupPipeline(PipelineState &state)
{
	RenderGraphBuilder builder;

	Renderer::instance()->buildRenderGraph(builder);
	auto &rg = builder.build();

	state.addOutputAttachment(m_output, m_attachment_state);
	state.declareAttachment(m_output, Graphics::RenderContext::GetSurface().GetFormat().format);

	for (auto &[name, output] : rg->getAttachments())
	{
		if (name != rg->output())
		{
			state.addDependency(name, VK_IMAGE_USAGE_SAMPLED_BIT);
		}
	}
}

void ImGuiPass::resolveResources(ResolveState &resolve)
{
}

void ImGuiPass::render(RenderPassState &state)
{
	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	vkCmdBeginRenderPass(state.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	ImGuiContext::render(state.command_buffer);

	vkCmdEndRenderPass(state.command_buffer);
}
}        // namespace Ilum::pass