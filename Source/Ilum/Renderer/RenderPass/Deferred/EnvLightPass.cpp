#include "EnvLightPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Scene/Component/Camera.hpp"

namespace Ilum::pass
{
EnvLightPass::EnvLightPass()
{
	// Setup a cube
	std::vector<glm::vec3> vertices = {
	    {-1.0f, -1.0f, -1.0f},        // bottom-left
	    {1.0f, 1.0f, -1.0f},          // top-right
	    {1.0f, -1.0f, -1.0f},         // bottom-right
	    {1.0f, 1.0f, -1.0f},          // top-right
	    {-1.0f, -1.0f, -1.0f},        // bottom-left
	    {-1.0f, 1.0f, -1.0f},         // top-left
	                                  // front face
	    {-1.0f, -1.0f, 1.0f},         // bottom-left
	    {1.0f, -1.0f, 1.0f},          // bottom-right
	    {1.0f, 1.0f, 1.0f},           // top-right
	    {1.0f, 1.0f, 1.0f},           // top-right
	    {-1.0f, 1.0f, 1.0f},          // top-left
	    {-1.0f, -1.0f, 1.0f},         // bottom-left
	                                  // left face
	    {-1.0f, 1.0f, 1.0f},          // top-right
	    {-1.0f, 1.0f, -1.0f},         // top-left
	    {-1.0f, -1.0f, -1.0f},        // bottom-left
	    {-1.0f, -1.0f, -1.0f},        // bottom-left
	    {-1.0f, -1.0f, 1.0f},         // bottom-right
	    {-1.0f, 1.0f, 1.0f},          // top-right
	                                  // right face
	    {1.0f, 1.0f, 1.0f},           // top-left
	    {1.0f, -1.0f, -1.0f},         // bottom-right
	    {1.0f, 1.0f, -1.0f},          // top-right
	    {1.0f, -1.0f, -1.0f},         // bottom-right
	    {1.0f, 1.0f, 1.0f},           // top-left
	    {1.0f, -1.0f, 1.0f},          // bottom-left
	                                  // bottom face
	    {-1.0f, -1.0f, -1.0f},        // top-right
	    {1.0f, -1.0f, -1.0f},         // top-left
	    {1.0f, -1.0f, 1.0f},          // bottom-left
	    {1.0f, -1.0f, 1.0f},          // bottom-left
	    {-1.0f, -1.0f, 1.0f},         // bottom-right
	    {-1.0f, -1.0f, -1.0f},        // top-right
	                                  // top face
	    {-1.0f, 1.0f, -1.0f},         // top-left
	    {1.0f, 1.0f, 1.0f},           // bottom-right
	    {1.0f, 1.0f, -1.0f},          // top-right
	    {1.0f, 1.0f, 1.0f},           // bottom-right
	    {-1.0f, 1.0f, -1.0f},         // top-left
	    {-1.0f, 1.0f, 1.0f},          // bottom-left
	};

	{
		Buffer staging_buffer(vertices.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		m_vertex_buffer = Buffer(vertices.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

		std::memcpy(staging_buffer.map(), vertices.data(), vertices.size() * sizeof(glm::vec3));
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.copyBuffer(BufferInfo{staging_buffer}, BufferInfo{m_vertex_buffer}, vertices.size() * sizeof(glm::vec3));
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}

void EnvLightPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/environment.vert", VK_SHADER_STAGE_VERTEX_BIT, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/environment.frag", VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Type::GLSL);

	state.dynamic_state.dynamic_states = {
	    VK_DYNAMIC_STATE_VIEWPORT,
	    VK_DYNAMIC_STATE_SCISSOR};

	state.vertex_input_state.attribute_descriptions = {
	    VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
	};

	state.vertex_input_state.binding_descriptions = {
	    VkVertexInputBindingDescription{0, sizeof(glm::vec3), VK_VERTEX_INPUT_RATE_VERTEX}};

	state.color_blend_attachment_states.resize(1);
	state.depth_stencil_state.stencil_test_enable = false;

	// Disable blending
	for (auto &color_blend_attachment_state : state.color_blend_attachment_states)
	{
		color_blend_attachment_state.blend_enable = false;
	}

	state.rasterization_state.polygon_mode = VK_POLYGON_MODE_FILL;

	state.descriptor_bindings.bind(0, 0, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 1, "generated_cubmap", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

	state.declareAttachment("lighting", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.declareAttachment("depth_stencil", VK_FORMAT_D32_SFLOAT_S8_UINT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);

	state.addOutputAttachment("lighting", AttachmentState::Load_Color);
	state.addOutputAttachment("depth_stencil", AttachmentState::Load_Depth_Stencil);
}

void EnvLightPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void EnvLightPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (!Renderer::instance()->hasMainCamera())
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

	VkDeviceSize offsets[1] = {0};
	vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &m_vertex_buffer.getBuffer(), offsets);

	vkCmdDraw(cmd_buffer, m_vertex_buffer.getSize() / sizeof(glm::vec3), 1, 0, 0);

	vkCmdEndRenderPass(cmd_buffer);
}
}        // namespace Ilum::pass