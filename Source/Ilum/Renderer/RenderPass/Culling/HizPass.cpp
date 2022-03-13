#include "HizPass.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "Device/LogicalDevice.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
HizPass::~HizPass()
{
	for (auto &view : m_views)
	{
		vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), view, nullptr);
	}
}

void HizPass::setupPipeline(PipelineState &state)
{
	state.declareAttachment("HizBuffer", VK_FORMAT_R32_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height, true);
	state.addOutputAttachment("HizBuffer", AttachmentState::Clear_Color);
	state.addDependency("HizBuffer", VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void HizPass::resolveResources(ResolveState &resolve)
{

}

void HizPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	auto &hiz = state.graph.getAttachment("HizBuffer");

	cmd_buffer.generateMipmaps(hiz, VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FILTER_NEAREST);

	if (m_views.empty())
	{
		VkImageViewCreateInfo view_create_info = {};
		view_create_info.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_create_info.image                 = hiz;
		view_create_info.viewType              = VK_IMAGE_VIEW_TYPE_2D;
		view_create_info.format                = hiz.getFormat();

		m_views.resize(hiz.getMipLevelCount());
		for (uint32_t i = 0; i < hiz.getMipLevelCount(); i++)
		{
			view_create_info.components = {
			    VK_COMPONENT_SWIZZLE_IDENTITY,
			    VK_COMPONENT_SWIZZLE_IDENTITY,
			    VK_COMPONENT_SWIZZLE_IDENTITY,
			    VK_COMPONENT_SWIZZLE_IDENTITY};
			view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
			view_create_info.subresourceRange.baseArrayLayer = 0;
			view_create_info.subresourceRange.layerCount     = 1;
			view_create_info.subresourceRange.baseMipLevel   = i;
			view_create_info.subresourceRange.levelCount     = 1;

			vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &m_views[i]);
		}
	}
}

void HizPass::onImGui()
{
	std::string items;
	for (size_t i = 0; i < m_views.size(); i++)
	{
		items += std::to_string(i) + '\0';
	}
	items += '\0';
	ImGui::Text("Hierarchy Z Buffer: ");
	ImGui::SameLine();
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Mip Level", &m_current_level, items.data());
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(m_views[m_current_level], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass