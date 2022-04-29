#include "IBL.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include "File/FileSystem.hpp"

#include "Device/LogicalDevice.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

namespace Ilum::pass
{
static bool Update = false;

EquirectangularToCubemap::~EquirectangularToCubemap()
{
	for (auto &framebuffer : m_framebuffers)
	{
		vkDestroyFramebuffer(GraphicsContext::instance()->getLogicalDevice(), framebuffer, nullptr);
	}
}

void EquirectangularToCubemap::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/EquirectangularToCubemap.hlsl", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::HLSL, "VSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/EquirectangularToCubemap.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;

	state.descriptor_bindings.bind(0, 0, "TextureArray", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Wrap), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("SkyBox", VK_FORMAT_R16G16B16A16_SFLOAT, 1024, 1024, false, 6);

	state.addOutputAttachment("SkyBox", AttachmentState::Clear_Color);
}

void EquirectangularToCubemap::resolveResources(ResolveState &resolve)
{
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
}

void EquirectangularToCubemap::render(RenderPassState &state)
{
	Renderer::instance()->Render_Stats.cubemap_update = false;

	if (!Update)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	auto &attachment = Renderer::instance()->getRenderGraph()->getAttachment("SkyBox");

	if (m_framebuffers.empty())
	{
		VK_Debugger::setName(attachment, "cubemap");

		m_framebuffers.resize(6);

		for (uint32_t layer = 0; layer < 6; layer++)
		{
			VkFramebufferCreateInfo frame_buffer_create_info = {};
			frame_buffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frame_buffer_create_info.renderPass              = state.pass.render_pass;
			frame_buffer_create_info.attachmentCount         = 1;
			frame_buffer_create_info.pAttachments            = &attachment.getView(layer);
			frame_buffer_create_info.width                   = 1024;
			frame_buffer_create_info.height                  = 1024;
			frame_buffer_create_info.layers                  = 1;

			vkCreateFramebuffer(GraphicsContext::instance()->getLogicalDevice(), &frame_buffer_create_info, nullptr, &m_framebuffers[layer]);
		}
	}

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	VkViewport viewport = {0, 0, 1024, 1024, 0, 1};
	VkRect2D   scissor  = {0, 0, 1024, 1024};

	glm::mat4 projection_matrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
	glm::mat4 views_matrix[] =
	    {
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
	        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

	m_push_data.tex_idx = Renderer::instance()->getResourceCache().imageID(m_filename);

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	for (uint32_t i = 0; i < 6; i++)
	{
		begin_info.framebuffer = m_framebuffers[i];
		vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		m_push_data.inverse_view_projection = glm::inverse(projection_matrix * views_matrix[i]);

		vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
		vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_push_data), &m_push_data);

		vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

		vkCmdEndRenderPass(cmd_buffer);
	}

	Update                                            = false;
	Renderer::instance()->Render_Stats.cubemap_update = true;
}

void EquirectangularToCubemap::onImGui()
{
	ImGui::PushID("Environment Light");
	if (ImGui::ImageButton(Renderer::instance()->getResourceCache().hasImage(FileSystem::getRelativePath(m_filename)) ?
	                           ImGuiContext::textureID(Renderer::instance()->getResourceCache().loadImage(FileSystem::getRelativePath(m_filename)), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)) :
                               ImGuiContext::textureID(Renderer::instance()->getDefaultTexture(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
	                       ImVec2{100.f, 100.f}))
	{
		m_filename = "";
		Update     = true;
	}
	ImGui::PopID();

	if (ImGui::BeginDragDropTarget())
	{
		if (const auto *pay_load = ImGui::AcceptDragDropPayload("Texture2D"))
		{
			ASSERT(pay_load->DataSize == sizeof(std::string));
			if (m_filename != *static_cast<std::string *>(pay_load->Data))
			{
				m_filename = *static_cast<std::string *>(pay_load->Data);
				Update     = true;
			}
		}
		ImGui::EndDragDropTarget();
	}
}

void BRDFPreIntegrate::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/BRDFPreIntegrate.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.declareAttachment("BRDFPreIntegrate", VK_FORMAT_R16G16_SFLOAT, 512, 512, false, 1);
	state.addOutputAttachment("BRDFPreIntegrate", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "BRDFPreIntegrate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BRDFPreIntegrate::resolveResources(ResolveState &resolve)
{
}

void BRDFPreIntegrate::render(RenderPassState &state)
{
	if (!m_finish)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, 512 / 8, 512 / 8, 1);

	m_finish = false;
}

void BRDFPreIntegrate::onImGui()
{
	const auto &brdf = Renderer::instance()->getRenderGraph()->getAttachment("BRDFPreIntegrate");

	ImGui::Text("BRDF PreIntegrate Result: ");
	ImGui::Image(ImGuiContext::textureID(brdf.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(100, 100));
}

void CubemapSHProjection::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapSHProjection.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.descriptor_bindings.bind(0, 1, "SkyBox", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Cube, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("SHIntermediate", VK_FORMAT_R16G16B16A16_SFLOAT, 1024 / 8 * 9, 1024 / 8, false, 6);
	state.addOutputAttachment("SHIntermediate", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "SHIntermediate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CubemapSHProjection::resolveResources(ResolveState &resolve)
{
}

void CubemapSHProjection::render(RenderPassState &state)
{
	if (!Update && !Renderer::instance()->Render_Stats.cubemap_update)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	VkExtent2D extent = {1024, 1024};

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extent), &extent);

	vkCmdDispatch(cmd_buffer, 1024 / 8, 1024 / 8, 6);

	Update = false;
}

void CubemapSHProjection::onImGui()
{
	if (ImGui::Button("Update"))
	{
		Update = true;
	}

	const auto &SHIntermediate = Renderer::instance()->getRenderGraph()->getAttachment("SHIntermediate");
	ImGui::Text("SHIntermediate Result: ");

	ImGui::PushItemWidth(100.f);
	ImGui::Combo("Face index", &m_face_id, "+X\0-X\0+Y\0-Y\0+Z\0-Z\0\0");
	ImGui::PopItemWidth();
	ImGui::Image(ImGuiContext::textureID(SHIntermediate.getView(m_face_id), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(300, 50));
}

void CubemapSHAdd::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapSHAdd.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.descriptor_bindings.bind(0, 1, "SHIntermediate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

	state.declareAttachment("IrradianceSH", VK_FORMAT_R16G16B16A16_SFLOAT, 9, 1, false, 1);
	state.addOutputAttachment("IrradianceSH", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "IrradianceSH", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void CubemapSHAdd::resolveResources(ResolveState &resolve)
{
}

void CubemapSHAdd::render(RenderPassState &state)
{
	if (!Update && !Renderer::instance()->Render_Stats.cubemap_update)
	{
		return;
	}

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, 9, 1, 1);

	Update = false;
}

void CubemapSHAdd::onImGui()
{
	if (ImGui::Button("Update"))
	{
		Update = true;
	}

	const auto &IrradianceSH = Renderer::instance()->getRenderGraph()->getAttachment("IrradianceSH");
	ImGui::Text("IrradianceSH Result: ");
	ImGui::Image(ImGuiContext::textureID(IrradianceSH.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(300, 50));
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
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PreProcess/CubemapPrefilter.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

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
	if (!Update && !Renderer::instance()->Render_Stats.cubemap_update)
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
		view_create_info.viewType              = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		view_create_info.format                = VK_FORMAT_R16G16B16A16_SFLOAT;
		view_create_info.components            = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY};

		VkDescriptorImageInfo image_info[2];

		image_info[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		image_info[0].sampler     = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);

		image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image_info[1].imageView   = skybox_map.getView(ImageViewType::Cube);
		image_info[1].sampler     = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);

		std::array<VkWriteDescriptorSet, 2> write_descriptor_sets;
		write_descriptor_sets[0]                  = {};
		write_descriptor_sets[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptor_sets[0].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		write_descriptor_sets[0].dstBinding       = 0;
		write_descriptor_sets[0].pImageInfo       = &image_info[0];
		write_descriptor_sets[0].pBufferInfo      = nullptr;
		write_descriptor_sets[0].pTexelBufferView = nullptr;
		write_descriptor_sets[0].descriptorCount  = 1;
		write_descriptor_sets[0].pNext            = nullptr;

		write_descriptor_sets[1]                  = {};
		write_descriptor_sets[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
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

	Update = false;
}

void CubemapPrefilter::onImGui()
{
	const auto &prefilter_map = Renderer::instance()->getRenderGraph()->getAttachment("PrefilterMap");

	if (ImGui::Button("Update"))
	{
		Update = true;
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
}