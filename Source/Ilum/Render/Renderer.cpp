#include "Renderer.hpp"

#include <RHI/Command.hpp>
#include <RHI/DescriptorState.hpp>
#include <RHI/PipelineState.hpp>

#include <imgui.h>

namespace Ilum
{
Renderer::Renderer(RHIDevice *device) :
    p_device(device),
    m_rg(device),
    m_rg_builder(device, m_rg)
{
	CreateSampler();
	KullaContyApprox();
	BRDFPreIntegration();
}

Renderer::~Renderer()
{
}

void Renderer::Tick()
{
	m_rg.Execute();
}

void Renderer::OnImGui(ImGuiContext &context)
{
	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	ImGui::Begin("Renderer");
	if (ImGui::TreeNode("LUT"))
	{
		if (ImGui::TreeNode("Kulla Conty Energy"))
		{
			ImGui::Image(context.TextureID(m_kulla_conty_EmuLut->GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("Kulla Conty Energy Average"))
		{
			ImGui::Image(context.TextureID(m_kulla_conty_EavgLut->GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		if (ImGui::TreeNode("BRDF PreIntegration"))
		{
			ImGui::Image(context.TextureID(m_brdf_preintegration->GetView(view_desc)), ImVec2(300, 300));
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Render Graph Nodes"))
	{
		m_rg.OnImGui(context);
		ImGui::TreePop();
	}

	m_rg_builder.OnImGui(context);


	ImGui::End();
}

void Renderer::CreateSampler()
{
	m_point_clamp_sampler       = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_point_warp_sampler        = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_bilinear_clamp_sampler    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_bilinear_warp_sampler     = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_NEAREST});
	m_trilinear_clamp_sampler   = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_trilinear_warp_sampler    = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_anisptropic_warp_sampler  = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_MIPMAP_MODE_LINEAR});
	m_anisptropic_clamp_sampler = std::make_unique<Sampler>(p_device, SamplerDesc{VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_MIPMAP_MODE_LINEAR});
}

void Renderer::KullaContyApprox()
{
	// Declare render target
	TextureDesc tex_desc  = {};
	tex_desc.width        = 1024;
	tex_desc.height       = 1024;
	tex_desc.depth        = 1;
	tex_desc.mips         = 1;
	tex_desc.layers       = 1;
	tex_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
	tex_desc.format       = VK_FORMAT_R16_SFLOAT;
	tex_desc.usage        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	m_kulla_conty_EmuLut  = std::make_unique<Texture>(p_device, tex_desc);
	m_kulla_conty_EavgLut = std::make_unique<Texture>(p_device, tex_desc);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Setup PSO
	ShaderDesc kulla_conty_energy_shader  = {};
	kulla_conty_energy_shader.filename    = "./Source/Shaders/Precompute/KullaContyEnergy.hlsl";
	kulla_conty_energy_shader.entry_point = "main";
	kulla_conty_energy_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	kulla_conty_energy_shader.type        = ShaderType::HLSL;

	ShaderDesc kulla_conty_average_shader  = {};
	kulla_conty_average_shader.filename    = "./Source/Shaders/Precompute/KullaContyEnergyAverage.hlsl";
	kulla_conty_average_shader.entry_point = "main";
	kulla_conty_average_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	kulla_conty_average_shader.type        = ShaderType::HLSL;

	PipelineState Emu_pso;
	Emu_pso.LoadShader(kulla_conty_energy_shader);

	PipelineState Eavg_pso;
	Eavg_pso.LoadShader(kulla_conty_average_shader);

	// Record command buffer
	auto &cmd_buffer = p_device->RequestCommandBuffer();

	cmd_buffer.Begin();

	// Comput Emu
	{
		cmd_buffer.Transition(m_kulla_conty_EmuLut.get(), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(Emu_pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, m_kulla_conty_EmuLut->GetView(view_desc)));
		cmd_buffer.Dispatch(1024 / 32, 1024 / 32);
		cmd_buffer.Transition(m_kulla_conty_EmuLut.get(), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	// Compute Eavg
	{
		cmd_buffer.Transition(m_kulla_conty_EavgLut.get(), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(Eavg_pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, m_kulla_conty_EavgLut->GetView(view_desc))
		        .Bind(0, 1, m_kulla_conty_EmuLut->GetView(view_desc)));
		cmd_buffer.Dispatch(1024 / 32, 1024 / 32);
		cmd_buffer.Transition(m_kulla_conty_EavgLut.get(), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

void Renderer::BRDFPreIntegration()
{
	// Declare render target
	TextureDesc tex_desc  = {};
	tex_desc.width        = 512;
	tex_desc.height       = 512;
	tex_desc.depth        = 1;
	tex_desc.mips         = 1;
	tex_desc.layers       = 1;
	tex_desc.sample_count = VK_SAMPLE_COUNT_1_BIT;
	tex_desc.format       = VK_FORMAT_R16G16_SFLOAT;
	tex_desc.usage        = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	m_brdf_preintegration = std::make_unique<Texture>(p_device, tex_desc);

	TextureViewDesc view_desc  = {};
	view_desc.aspect           = VK_IMAGE_ASPECT_COLOR_BIT;
	view_desc.view_type        = VK_IMAGE_VIEW_TYPE_2D;
	view_desc.base_array_layer = 0;
	view_desc.base_mip_level   = 0;
	view_desc.layer_count      = 1;
	view_desc.level_count      = 1;

	// Setup PSO
	ShaderDesc brdf_preintegration_shader  = {};
	brdf_preintegration_shader.filename    = "./Source/Shaders/Precompute/BRDFPreIntegration.hlsl";
	brdf_preintegration_shader.entry_point = "main";
	brdf_preintegration_shader.stage       = VK_SHADER_STAGE_COMPUTE_BIT;
	brdf_preintegration_shader.type        = ShaderType::HLSL;

	PipelineState pso;
	pso.LoadShader(brdf_preintegration_shader);

	// Record command buffer
	auto &cmd_buffer = p_device->RequestCommandBuffer();

	cmd_buffer.Begin();

	// Comput BRDF Preintegration
	{
		cmd_buffer.Transition(m_brdf_preintegration.get(), TextureState{}, TextureState(VK_IMAGE_USAGE_STORAGE_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
		cmd_buffer.Bind(pso);
		cmd_buffer.Bind(
		    cmd_buffer.GetDescriptorState()
		        .Bind(0, 0, m_brdf_preintegration->GetView(view_desc)));
		cmd_buffer.Dispatch(512 / 32, 512 / 32);
		cmd_buffer.Transition(m_brdf_preintegration.get(), TextureState{VK_IMAGE_USAGE_STORAGE_BIT}, TextureState(VK_IMAGE_USAGE_SAMPLED_BIT), VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
	}

	cmd_buffer.End();

	// Submit
	p_device->SubmitIdle(cmd_buffer);
}

}        // namespace Ilum