#include "LightPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/Tag.hpp"
#include "Scene/Component/Transform.hpp"
#include "Scene/Scene.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include <entt.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

namespace Ilum::pass
{
LightPass::LightPass()
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.minFilter           = VK_FILTER_LINEAR;
	create_info.magFilter           = VK_FILTER_LINEAR;
	create_info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	create_info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	create_info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	create_info.mipLodBias          = 0;
	create_info.minLod              = 0;
	create_info.maxLod              = 1000;
	create_info.borderColor         = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

	VkSampler sampler;
	vkCreateSampler(GraphicsContext::instance()->getLogicalDevice(), &create_info, nullptr, &sampler);
	m_shadowmap_sampler = Sampler(sampler);
}

void LightPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Deferred/Lighting.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.color_blend_attachment_states[0].blend_enable = false;

	state.descriptor_bindings.bind(0, 0, "GBuffer0", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "GBuffer1", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 2, "GBuffer2", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 3, "GBuffer3", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 4, "GBuffer4", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 5, "GBuffer5", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 6, "DepthStencil", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Depth_Only, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 7, "EmuLut", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 8, "EavgLut", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 9, "Shadowmap", m_shadowmap_sampler, ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 10, "CascadeShadowmap", m_shadowmap_sampler, ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 11, "DirectionalLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 12, "PointLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 13, "SpotLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 14, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("Lighting", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("Lighting", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 15, "Lighting", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void LightPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("DirectionalLights", Renderer::instance()->Render_Buffer.Directional_Light_Buffer);
	resolve.resolve("PointLights", Renderer::instance()->Render_Buffer.Point_Light_Buffer);
	resolve.resolve("SpotLights", Renderer::instance()->Render_Buffer.Spot_Light_Buffer);
}

void LightPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (!Renderer::instance()->hasMainCamera())
	{
		return;
	}

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	m_push_block.directional_light_count = Renderer::instance()->Render_Stats.light_count.directional_light_count;
	m_push_block.spot_light_count        = Renderer::instance()->Render_Stats.light_count.spot_light_count;
	m_push_block.point_light_count       = Renderer::instance()->Render_Stats.light_count.point_light_count;
	m_push_block.extent                  = Renderer::instance()->getRenderTargetExtent();

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_block), &m_push_block);
	vkCmdDispatch(cmd_buffer, (m_push_block.extent.width + 32 - 1) / 32, (m_push_block.extent.height + 32 - 1) / 32, 1);
}

void LightPass::onImGui()
{
	ImGui::Checkbox("Enable Kulla Conty Multi-Bounce Approximation", reinterpret_cast<bool *>(&m_push_block.enable_multi_bounce));
	if (ImGui::TreeNode("Soft Shadow"))
	{
		const char *const filter_method[] = {"None", "PCF", "PCSS"};
		ImGui::Combo("Filter method", &m_push_block.filter_method, filter_method, 3);
		if (m_push_block.filter_method == 1 || m_push_block.filter_method == 2)
		{
			ImGui::DragInt("Number of Samples", &m_push_block.sample_num, 0.1f, 0);
			ImGui::DragFloat("Filter scale", &m_push_block.sample_scale, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
			if (m_push_block.filter_method == 2)
			{
				ImGui::DragFloat("Light size", &m_push_block.light_size, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
			}

			const char *const sample_method[] = {"Uniform", "Possion"};
			ImGui::Combo("Sample method", &m_push_block.sample_method, sample_method, 2);
		}

		ImGui::TreePop();
	}
}
}        // namespace Ilum::pass