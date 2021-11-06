#include "ImGuiPass.hpp"

#include "Device/Instance.hpp"
#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"
#include "Device/Surface.hpp"
#include "Device/Swapchain.hpp"
#include "Device/Window.hpp"

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

	for (auto &[name, output] : rg->getAttachments())
	{
		if (name != rg->output())
		{
			state.addDependency(name, VK_IMAGE_USAGE_SAMPLED_BIT);
		}
	}

	state.addOutputAttachment(m_output, m_attachment_state);
	state.declareAttachment(m_output, GraphicsContext::instance()->getSurface().getFormat().format);
	state.addDependency(m_view, VK_IMAGE_USAGE_SAMPLED_BIT);
}

void ImGuiPass::resolveResources(ResolveState &resolve)
{
}

void ImGuiPass::render(RenderPassState &state)
{
	ImGuiContext::render(state.command_buffer);
}
}        // namespace Ilum::pass