#include "VisualizeVBuffer.hpp"

#include <RHI/DescriptorState.hpp>
#include <RHI/FrameBuffer.hpp>

#include <Render/RGBuilder.hpp>
#include <Render/Renderer.hpp>

namespace Ilum
{
VisualizeVBuffer::VisualizeVBuffer() :
    RenderPass("VisualizeVBuffer")
{
}

void VisualizeVBuffer::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<VisualizeVBuffer>();

	// Render Target
	auto vbuffer = builder.CreateTexture(
	    "Visibility Buffer",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R32_UINT,
	        VK_IMAGE_USAGE_SAMPLED_BIT},
	    TextureState{VK_IMAGE_USAGE_SAMPLED_BIT});

	auto instance = builder.CreateTexture(
	    "Instance",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R8G8B8A8_UNORM,
	        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	auto primitive = builder.CreateTexture(
	    "Primitive",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R8G8B8A8_UNORM,
	        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	auto meshlet = builder.CreateTexture(
	    "Meshlet",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R8G8B8A8_UNORM,
	        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	pass->AddResource(vbuffer);
	pass->AddResource(instance);
	pass->AddResource(primitive);
	pass->AddResource(meshlet);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc shader  = {};
	shader.filename    = "./Source/Shaders/VisualizeVBuffer.hlsl";
	shader.entry_point = "main";
	shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso
	    .SetName("VisualizeVBuffer")
	    .LoadShader(shader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(vbuffer)->GetView(view_desc))
		        .Bind(0, 1, resource.GetTexture(instance)->GetView(view_desc))
		        .Bind(0, 2, resource.GetTexture(primitive)->GetView(view_desc))
		        .Bind(0, 3, resource.GetTexture(meshlet)->GetView(view_desc)));
		cmd_buffer.Bind(pso);
		cmd_buffer.Dispatch((renderer.GetExtent().width + 32 - 1) / 32, (renderer.GetExtent().height + 32 - 1) / 32);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {

	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum