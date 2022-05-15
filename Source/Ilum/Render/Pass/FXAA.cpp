#include "FXAA.hpp"

#include <RHI/DescriptorState.hpp>

#include "Render/RGBuilder.hpp"
#include "Render/Renderer.hpp"

#include <imgui.h>

namespace Ilum
{
FXAA::FXAA() :
    RenderPass("FXAA")
{
}

void FXAA::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<FXAA>();

	auto input = builder.CreateTexture(
	    "Input",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R16G16B16A16_SFLOAT,
	        VK_IMAGE_USAGE_SAMPLED_BIT},
	    TextureState{VK_IMAGE_USAGE_SAMPLED_BIT});

	auto output = builder.CreateTexture(
	    "Output",
	    TextureDesc{
	        builder.GetRenderer().GetExtent().width,  /*width*/
	        builder.GetRenderer().GetExtent().height, /*height*/
	        1,                                        /*depth*/
	        1,                                        /*mips*/
	        1,                                        /*layers*/
	        VK_SAMPLE_COUNT_1_BIT,
	        VK_FORMAT_R16G16B16A16_SFLOAT,
	        VK_IMAGE_USAGE_STORAGE_BIT},
	    TextureState{VK_IMAGE_USAGE_STORAGE_BIT});

	pass->AddResource(input);
	pass->AddResource(output);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ShaderDesc fxaa_hader  = {};
	fxaa_hader.filename    = "./Source/Shaders/FXAA.hlsl";
	fxaa_hader.entry_point = "main";
	fxaa_hader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	fxaa_hader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso
	    .SetName("FXAA")
	    .LoadShader(fxaa_hader);

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(input)->GetView(view_desc))
		        .Bind(0, 1, renderer.GetSampler(SamplerType::TrilinearClamp))
		        .Bind(0, 2, resource.GetTexture(output)->GetView(view_desc)));
		cmd_buffer.Dispatch((renderer.GetExtent().width + 8 - 1) / 8, (renderer.GetExtent().height + 8 - 1) / 8);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {
	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum