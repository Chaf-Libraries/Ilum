#include "Tonemapping.hpp"

#include <RHI/DescriptorState.hpp>

#include "Render/RGBuilder.hpp"
#include "Render/Renderer.hpp"

#include <imgui.h>

namespace Ilum
{
enum class TonemapType
{
	Uncharted,
	Hejlrichard,
	ACES
};

Tonemapping::Tonemapping() :
    RenderPass("Tonemapping")
{
}

void Tonemapping::Create(RGBuilder &builder)
{
	std::unique_ptr<RenderPass> pass = std::make_unique<Tonemapping>();

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
	        VK_FORMAT_R8G8B8A8_UNORM,
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

	ShaderDesc uncharted_shader  = {};
	uncharted_shader.filename    = "./Source/Shaders/Tonemapping.hlsl";
	uncharted_shader.entry_point = "main";
	uncharted_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	uncharted_shader.type        = ShaderType::HLSL;
	uncharted_shader.macros.push_back("TONEMAP_UNCHARTED");

	ShaderDesc hejlrichard_shader  = {};
	hejlrichard_shader.filename    = "./Source/Shaders/Tonemapping.hlsl";
	hejlrichard_shader.entry_point = "main";
	hejlrichard_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	hejlrichard_shader.type        = ShaderType::HLSL;
	hejlrichard_shader.macros.push_back("TONEMAP_HEJLRICHARD");

	ShaderDesc aces_shader  = {};
	aces_shader.filename    = "./Source/Shaders/Tonemapping.hlsl";
	aces_shader.entry_point = "main";
	aces_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	aces_shader.type        = ShaderType::HLSL;
	aces_shader.macros.push_back("TONEMAP_ACES");

	PipelineState uncharted_pso;
	uncharted_pso
	    .SetName("Tonemapping")
	    .LoadShader(uncharted_shader);

	PipelineState hejlrichard_pso;
	hejlrichard_pso
	    .SetName("Tonemapping")
	    .LoadShader(hejlrichard_shader);

	PipelineState aces_pso;
	aces_pso
	    .SetName("Tonemapping")
	    .LoadShader(aces_shader);

	struct PushConstant
	{
		struct Data
		{
			float   brightness   = 1.f;
			float   contrast     = 1.f;
			float   saturation   = 1.f;
			float   vignette     = 0.f;
			float   avgLum       = 1.f;
			int32_t autoExposure = 0;
			float   Ywhite       = 0.5f;        // Burning white
			float   key          = 0.5f;        // Log-average luminance
		}data;
		TonemapType type = TonemapType::Uncharted;
	};
	std::shared_ptr<PushConstant> m_push_constant = std::make_unique<PushConstant>();

	pass->BindCallback([=](CommandBuffer &cmd_buffer, const RGResources &resource, Renderer &renderer) {
		std::shared_ptr<PushConstant> push_constant = m_push_constant;
		switch (m_push_constant->type)
		{
			case TonemapType::Uncharted:
				cmd_buffer.Bind(uncharted_pso);
				break;
			case TonemapType::Hejlrichard:
				cmd_buffer.Bind(hejlrichard_pso);
				break;
			case TonemapType::ACES:
				cmd_buffer.Bind(aces_pso);
				break;
			default:
				cmd_buffer.Bind(uncharted_pso);
				break;
		}
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, resource.GetTexture(input)->GetView(view_desc))
		        .Bind(0, 1, renderer.GetSampler(SamplerType::TrilinearClamp))
		        .Bind(0, 2, resource.GetTexture(output)->GetView(view_desc)));
		cmd_buffer.PushConstants(VK_SHADER_STAGE_COMPUTE_BIT, push_constant.get(), sizeof(PushConstant::Data), 0);
		cmd_buffer.Dispatch((renderer.GetExtent().width + 32 - 1) / 32, (renderer.GetExtent().height + 32 - 1) / 32);
	});

	pass->BindImGui([=](ImGuiContext &, const RGResources &) {
		std::shared_ptr<PushConstant> push_constant = m_push_constant;
		std::bitset<8>                b(push_constant->data.autoExposure);
		bool                          autoExposure = b.test(0);

		const char *const tonemap_type[] = {"Uncharted", "Hejlrichard", "ACES"};
		ImGui::Combo("Type", reinterpret_cast<int32_t *>(&push_constant->type), tonemap_type, 3);

		ImGui::Checkbox("Auto Exposure", &autoExposure);
		ImGui::SliderFloat("Exposure", &push_constant->data.avgLum, 0.001f, 5.0f, "%.3f");
		ImGui::SliderFloat("Brightness", &push_constant->data.brightness, 0.0f, 2.0f, "%.3f");
		ImGui::SliderFloat("Contrast", &push_constant->data.contrast, 0.0f, 2.0f, "%.3f");
		ImGui::SliderFloat("Saturation", &push_constant->data.saturation, 0.0f, 5.0f, "%.3f");
		ImGui::SliderFloat("Vignette", &push_constant->data.vignette, 0.0f, 2.0f, "%.3f");

		if (autoExposure)
		{
			bool localExposure = b.test(1);
			if (ImGui::TreeNode("Auto Settings"))
			{
				ImGui::Checkbox("Local", &localExposure);
				ImGui::SliderFloat("Burning White", &push_constant->data.Ywhite, 0.f, 1.f, "%.3f");
				ImGui::SliderFloat("Brightness", &push_constant->data.key, 0.f, 1.f, "%.3f");
				b.set(1, localExposure);
				ImGui::End();
			}
		}

		b.set(0, autoExposure);
		push_constant->data.autoExposure = b.to_ulong();
	});

	builder.AddPass(std::move(pass));
}

}        // namespace Ilum