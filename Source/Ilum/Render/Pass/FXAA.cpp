#include "FXAA.hpp"

#include <RHI/DescriptorState.hpp>

#include "Render/RGBuilder.hpp"
#include "Render/Renderer.hpp"

#include <imgui.h>

namespace Ilum
{
enum class FXAAQuality
{
	High,
	Medium,
	Low
};

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

	ShaderDesc high_shader  = {};
	high_shader.filename    = "./Source/Shaders/FXAA.hlsl";
	high_shader.entry_point = "main";
	high_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	high_shader.type        = ShaderType::HLSL;
	high_shader.macros.push_back("FXAA_QUALITY_HIGH");

	ShaderDesc medium_shader  = {};
	medium_shader.filename    = "./Source/Shaders/FXAA.hlsl";
	medium_shader.entry_point = "main";
	medium_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	medium_shader.type        = ShaderType::HLSL;
	medium_shader.macros.push_back("FXAA_QUALITY_MEDIUM");

	ShaderDesc low_shader  = {};
	low_shader.filename    = "./Source/Shaders/FXAA.hlsl";
	low_shader.entry_point = "main";
	low_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	low_shader.type        = ShaderType::HLSL;
	low_shader.macros.push_back("FXAA_QUALITY_LOW");

	PipelineState high_pso;
	high_pso
	    .SetName("FXAA")
	    .LoadShader(high_shader);

	PipelineState medium_pso;
	medium_pso
	    .SetName("FXAA")
	    .LoadShader(medium_shader);

	PipelineState low_pso;
	low_pso
	    .SetName("FXAA")
	    .LoadShader(low_shader);

	struct PushConstant
	{
		struct Data
		{
			float fixed_threshold    = 0.0383f;
			float relative_threshold = 0.1180f;
			float subpixel_blending  = 0.01f;
		} data;
		FXAAQuality quality = FXAAQuality::High;
	};
	std::shared_ptr<PushConstant> m_push_constant = std::make_unique<PushConstant>();

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		std::shared_ptr<PushConstant> push_constant = m_push_constant;
		switch (m_push_constant->quality)
		{
			case FXAAQuality::High:
				cmd_buffer.Bind(high_pso);
				break;
			case FXAAQuality::Medium:
				cmd_buffer.Bind(medium_pso);
				break;
			case FXAAQuality::Low:
				cmd_buffer.Bind(low_pso);
				break;
			default:
				cmd_buffer.Bind(high_pso);
				break;
		}
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(input)->GetView(view_desc))
		        .Bind(0, 1, renderer.GetSampler(SamplerType::TrilinearClamp))
		        .Bind(0, 2, resource.GetTexture(output)->GetView(view_desc)));
		cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, push_constant.get(), sizeof(PushConstant::Data), 0);
		cmd_buffer.Dispatch((renderer.GetExtent().width + 8 - 1) / 8, (renderer.GetExtent().height + 8 - 1) / 8);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {
		std::shared_ptr<PushConstant> push_constant = m_push_constant;
		const char *const             fxaa_quality[] = {"High", "Medium", "Low"};
		ImGui::Combo("Quality", reinterpret_cast<int32_t *>(&push_constant->quality), fxaa_quality, 3);
		ImGui::SliderFloat("Fixed Threshold", &m_push_constant->data.fixed_threshold, 0.0312f, 0.0833f, "%.4f");
		ImGui::SliderFloat("Relative Threshold", &m_push_constant->data.relative_threshold, 0.063f, 0.333f, "%.4f");
		ImGui::SliderFloat("Subpixel Blending", &m_push_constant->data.subpixel_blending, 0.f, 1.f, "%.2f");
	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum