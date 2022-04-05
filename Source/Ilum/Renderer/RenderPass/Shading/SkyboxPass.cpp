#include "SkyboxPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"

#include <imgui.h>

namespace Ilum::pass
{
void SkyboxPass::setupPipeline(PipelineState &state)
{
	//state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Skybox.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	//state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Skybox.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Skybox.hlsl", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::HLSL, "VSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Shading/Skybox.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");

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

	state.depth_stencil_state.depth_compare_op = VK_COMPARE_OP_LESS_OR_EQUAL;

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 1, "SkyBox", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Cube, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("Lighting", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("DepthStencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("Lighting", AttachmentState::Load_Color);
	state.addOutputAttachment("DepthStencil", AttachmentState::Load_Depth_Stencil);
}

void SkyboxPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void SkyboxPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (m_type == EnvLightType::None || !Renderer::instance()->hasMainCamera())
	{
		return;
	}

	VkRenderPassBeginInfo begin_info = {};
	begin_info.sType                 = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	begin_info.renderPass            = state.pass.render_pass;
	begin_info.renderArea            = state.pass.render_area;
	begin_info.framebuffer           = state.pass.frame_buffer;
	begin_info.clearValueCount       = static_cast<uint32_t>(state.pass.clear_values.size());
	begin_info.pClearValues          = state.pass.clear_values.data();

	vkCmdBeginRenderPass(cmd_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	VkViewport viewport = {0, static_cast<float>(extent.height), static_cast<float>(extent.width), -static_cast<float>(extent.height), 0, 1};
	VkRect2D   scissor  = {0, 0, extent.width, extent.height};

	vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
	vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

	vkCmdDraw(cmd_buffer, 36, 1, 0, 0);

	vkCmdEndRenderPass(cmd_buffer);
}

void SkyboxPass::onImGui()
{
	const char *const skybox_option[] = {"None", "HDRI"};
	ImGui::Combo("Skybox Option", reinterpret_cast<int *>(&m_type), skybox_option, 2);
}
}        // namespace Ilum::pass