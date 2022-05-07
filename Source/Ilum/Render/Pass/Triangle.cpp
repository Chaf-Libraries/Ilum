#include "Triangle.hpp"

#include "../RGBuilder.hpp"
#include "../RenderGraph.hpp"

#include <RHI/FrameBuffer.hpp>

namespace Ilum
{
Triangle::Triangle() :
    RenderPass("Triangle")
{
}

void Triangle::Prepare(PipelineState &pso)
{
	ShaderDesc vertex_shader  = {};
	vertex_shader.filename    = "./Source/Shaders/triangle.vert";
	vertex_shader.entry_point = "main";
	vertex_shader.stage       = VK_SHADER_STAGE_VERTEX_BIT;
	vertex_shader.type        = ShaderType::GLSL;

	ShaderDesc frag_shader  = {};
	frag_shader.filename    = "./Source/Shaders/triangle.frag";
	frag_shader.entry_point = "main";
	frag_shader.stage       = VK_SHADER_STAGE_FRAGMENT_BIT;
	frag_shader.type        = ShaderType::GLSL;

	DynamicState dynamic_state = {};
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_SCISSOR);
	dynamic_state.dynamic_states.push_back(VK_DYNAMIC_STATE_VIEWPORT);

	ColorBlendState color_blend_state = {};
	color_blend_state.attachment_states.push_back(ColorBlendAttachmentState{});

	pso.SetDynamicState(dynamic_state);
	pso.SetColorBlendState(color_blend_state);

	pso.LoadShader(vertex_shader);
	pso.LoadShader(frag_shader);
}

void Triangle::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<Triangle>();

	auto texture = builder.CreateTexture(
	    "Result",
	    TextureDesc{500, 500, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
	    TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

	pass->AddResource(texture);

	TextureDesc rt_desc = {};
	rt_desc.width       = 500;
	rt_desc.height      = 500;
	rt_desc.depth       = 1;
	rt_desc.mips        = 1;
	rt_desc.layers      = 1;
	rt_desc.format      = VK_FORMAT_R8G8B8A8_UNORM;
	rt_desc.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	pass->BindCallback([=](CommandBuffer &cmd_buffer, PipelineState &pso, const RGResources &resource, Renderer &) {
		FrameBuffer framebuffer;
		framebuffer.Bind(resource.GetTexture(texture), view_desc, ColorAttachmentInfo{});
		cmd_buffer.BeginRenderPass(framebuffer);
		cmd_buffer.Bind(pso);
		VkViewport viewport = {0, 0, 500, 500, 0, 1};
		VkRect2D   scissor  = {0, 0, 500, 500};
		cmd_buffer.SetViewport(500, 500);
		cmd_buffer.SetScissor(500, 500);
		vkCmdDraw(cmd_buffer, 3, 1, 0, 0);
		cmd_buffer.EndRenderPass();
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {
		
		});

	builder.AddPass(std::move(pass));
}
}        // namespace Ilum