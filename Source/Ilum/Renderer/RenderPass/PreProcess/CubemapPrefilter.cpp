#include "CubemapPrefilter.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include "ImGui/ImGuiContext.hpp"

#include <imgui.h>

namespace Ilum::pass
{
CubemapPrefilter::CubemapPrefilter()
{
}

CubemapPrefilter::~CubemapPrefilter()
{
	for (auto &view : m_views)
	{
		vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), view, nullptr);
	}
}

void CubemapPrefilter::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapPrefilter.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 1, "SkyBox", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Cube, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("PrefilterMap", VK_FORMAT_R16G16B16A16_SFLOAT, 512, 512, true, 6);
	state.addOutputAttachment("PrefilterMap", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "PrefilterMap", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

	for (uint32_t level = 0; level < m_mip_levels; level++)
	{
		m_descriptor_sets.push_back(DescriptorSet(state.shader));
	}
}

void CubemapPrefilter::resolveResources(ResolveState &resolve)
{
}

void CubemapPrefilter::render(RenderPassState &state)
{
	if (!m_update)
	{
		return;
	}

	// Create views and descriptor sets
	if (m_views.empty())
	{
		const auto &prefilter_map = Renderer::instance()->getRenderGraph()->getAttachment("PrefilterMap");
		const auto &skybox_map    = Renderer::instance()->getRenderGraph()->getAttachment("SkyBox");

		m_views.resize(m_mip_levels);

		VkImageViewCreateInfo view_create_info = {};
		view_create_info.sType                 = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		view_create_info.image                 = prefilter_map;
		view_create_info.viewType              = VK_IMAGE_VIEW_TYPE_CUBE;
		view_create_info.format                = VK_FORMAT_R16G16B16A16_SFLOAT;
		view_create_info.components            = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY};

		VkDescriptorImageInfo image_info[2];

		image_info[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		//image_info[1].imageView   = m_views;
		image_info[0].sampler = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);

		image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image_info[1].imageView   = skybox_map.getView(ImageViewType::Cube);
		image_info[1].sampler     = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);

		std::array<VkWriteDescriptorSet, 2> write_descriptor_sets;
		write_descriptor_sets[0]       = {};
		write_descriptor_sets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		//write_descriptor_sets[0].dstSet           = m_descriptor_sets[level];
		write_descriptor_sets[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		write_descriptor_sets[0].dstBinding       = 0;
		write_descriptor_sets[0].pImageInfo       = &image_info[0];
		write_descriptor_sets[0].pBufferInfo      = nullptr;
		write_descriptor_sets[0].pTexelBufferView = nullptr;
		write_descriptor_sets[0].descriptorCount  = 1;
		write_descriptor_sets[0].pNext            = nullptr;

		write_descriptor_sets[1]       = {};
		write_descriptor_sets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		//write_descriptor_sets[1].dstSet           = m_descriptor_sets[level];
		write_descriptor_sets[1].descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write_descriptor_sets[1].dstBinding       = 1;
		write_descriptor_sets[1].pImageInfo       = &image_info[1];
		write_descriptor_sets[1].pBufferInfo      = nullptr;
		write_descriptor_sets[1].pTexelBufferView = nullptr;
		write_descriptor_sets[1].descriptorCount  = 1;
		write_descriptor_sets[1].pNext            = nullptr;

		for (uint32_t i = 0; i < m_mip_levels; i++)
		{
			view_create_info.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 6};
			vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &view_create_info, nullptr, &m_views[i]);

			image_info[0].imageView = m_views[i];

			write_descriptor_sets[0].dstSet = m_descriptor_sets[i];
			write_descriptor_sets[1].dstSet = m_descriptor_sets[i];
			vkUpdateDescriptorSets(GraphicsContext::instance()->getLogicalDevice(), 2, write_descriptor_sets.data(), 0, nullptr);
		}
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (uint32_t i = 0; i < m_mip_levels; i++)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, 0, 1, &m_descriptor_sets[i], 0, nullptr);

		m_push_data.mip_extent = VkExtent2D{512u >> i, 512u >> i};
		m_push_data.roughness  = static_cast<float>(i) / (static_cast<float>(m_mip_levels - 1));

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_data), &m_push_data);

		vkCmdDispatch(cmd_buffer, m_push_data.mip_extent.width / 8, m_push_data.mip_extent.height / 8, 6);
	}

	m_update = false;
}

void CubemapPrefilter::onImGui()
{
	const auto &prefilter_map = Renderer::instance()->getRenderGraph()->getAttachment("PrefilterMap");

	if (ImGui::Button("Update"))
	{
		m_update = true;
	}

	std::string items;
	for (size_t i = 0; i < m_views.size(); i++)
	{
		items += std::to_string(i) + '\0';
	}
	items += '\0';
	ImGui::Text("Cubemap Prefilter: ");
	ImGui::SameLine();
	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Mip Level", &m_current_level, items.data());
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(prefilter_map.getView(m_current_level), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}
}        // namespace Ilum::pass