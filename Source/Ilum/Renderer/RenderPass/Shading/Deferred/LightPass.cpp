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
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/GLSL/Shading/Deferred/Lighting.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);
	//state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Deferred/Lighting.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.color_blend_attachment_states[0].blend_enable = false;

	state.descriptor_bindings.bind(0, 0, "texSampler", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), VK_DESCRIPTOR_TYPE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "shadowmapSampler", m_shadowmap_sampler, VK_DESCRIPTOR_TYPE_SAMPLER);
	state.descriptor_bindings.bind(0, 2, "GBuffer0", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 3, "GBuffer1", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 4, "GBuffer2", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 5, "GBuffer3", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 6, "GBuffer4", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 7, "GBuffer5", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 8, "DepthStencil", ImageViewType::Depth_Only, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 9, "EmuLut", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 10, "EavgLut", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 11, "Shadowmap", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 12, "CascadeShadowmap", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 13, "OmniShadowmap", ImageViewType::ArrayCube, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 14, "DirectionalLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 15, "PointLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 16, "SpotLights", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 17, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 18, "IrradianceSH", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 19, "PrefilterMap", ImageViewType::Cube, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 20, "BRDFPreIntegrate", ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

	state.declareAttachment("Lighting", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("Lighting", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 21, "Lighting", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void LightPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
	resolve.resolve("TextureArray", Renderer::instance()->getResourceCache().getImageReferences());
	resolve.resolve("DirectionalLights", Renderer::instance()->Render_Buffer.Directional_Light_Buffer);
	resolve.resolve("PointLights", Renderer::instance()->Render_Buffer.Point_Light_Buffer);
	resolve.resolve("SpotLights", Renderer::instance()->Render_Buffer.Spot_Light_Buffer);
	resolve.resolve("MaterialBuffer", Renderer::instance()->Render_Buffer.Material_Buffer);
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
}
}        // namespace Ilum::pass