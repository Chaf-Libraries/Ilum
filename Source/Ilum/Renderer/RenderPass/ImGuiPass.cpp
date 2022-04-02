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

	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/ImGui.hlsl", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::HLSL, "VSmain");
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/ImGui.hlsl", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::HLSL, "PSmain");
	//state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/ImGui.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	//state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/ImGui.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
	    VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32_SFLOAT, 0},
	    VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32B32A32_SFLOAT}};

	state.vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(float) * 8, VK_VERTEX_INPUT_RATE_VERTEX}};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.addOutputAttachment(m_output, m_attachment_state);
	state.declareAttachment(m_output, GraphicsContext::instance()->getSurface().getFormat().format);

	for (auto &[name, output] : rg->getAttachments())
	{
		if (name != m_output)
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