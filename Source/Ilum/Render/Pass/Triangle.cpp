#include "Triangle.hpp"

#include "../RGBuilder.hpp"
#include "../RenderGraph.hpp"
#include "../Renderer.hpp"

#include <RHI/FrameBuffer.hpp>

namespace Ilum
{
Triangle::Triangle() :
    RenderPass("Triangle")
{
	
}

void Triangle::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<Triangle>();

	TextureDesc rt_desc = {};
	rt_desc.width       = builder.GetRenderer().GetExtent().width;
	rt_desc.height      = builder.GetRenderer().GetExtent().height;
	rt_desc.depth       = 1;
	rt_desc.mips        = 1;
	rt_desc.layers      = 1;
	rt_desc.format      = VK_FORMAT_R8G8B8A8_UNORM;
	rt_desc.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	auto texture = builder.CreateTexture(
	    "Result",
	    rt_desc,
	    TextureState{VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT});

	pass->AddResource(texture);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

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

	PipelineState pso;
	pso.SetDynamicState(dynamic_state);
	pso.SetColorBlendState(color_blend_state);

	pso.LoadShader(vertex_shader);
	pso.LoadShader(frag_shader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		FrameBuffer framebuffer;
		framebuffer.Bind(resource.GetTexture(texture), view_desc, ColorAttachmentInfo{});
		cmd_buffer.BeginRenderPass(framebuffer);
		cmd_buffer.Bind(pso);
		cmd_buffer.SetViewport(static_cast<float>(renderer.GetViewport().width), static_cast<float>(renderer.GetViewport().height));
		cmd_buffer.SetScissor(renderer.GetViewport().width, renderer.GetViewport().height);
		vkCmdDraw(cmd_buffer, 3, 1, 0, 0);
		cmd_buffer.EndRenderPass();
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}
}        // namespace Ilum