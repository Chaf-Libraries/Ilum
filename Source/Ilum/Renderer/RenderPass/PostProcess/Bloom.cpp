#include "Bloom.hpp"

#include "Renderer/RenderGraph/RenderGraph.hpp"
#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"

#include "Device/LogicalDevice.hpp"

#include <imgui.h>

static bool  Enable_Bloom = true;
static float Threshold    = 0.75f;
static float Radius       = 0.75f;
static float Intensity    = 1.f;

namespace Ilum::pass
{
BloomMask::BloomMask(const std::string &input) :
    m_input(input)
{
}

void BloomMask::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomMask.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	state.descriptor_bindings.bind(0, 0, m_input, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

	state.declareAttachment("BloomMask", VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height);
	state.addOutputAttachment("BloomMask", AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, "BloomMask", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomMask::resolveResources(ResolveState &resolve)
{
}

void BloomMask::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (Enable_Bloom)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		auto &extent = Renderer::instance()->getRenderTargetExtent();

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &Threshold);
		vkCmdDispatch(cmd_buffer, (extent.width + 32 - 1) / 32, (extent.height + 32 - 1) / 32, 1);
	}
	else
	{
		VkClearColorValue clear_color = {};
		clear_color.float32[0]        = 0.f;
		clear_color.float32[1]        = 0.f;
		clear_color.float32[2]        = 0.f;
		clear_color.float32[3]        = 0.f;
		vkCmdClearColorImage(cmd_buffer, state.graph.getAttachment("BloomMask"), VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &state.graph.getAttachment("BloomMask").getSubresourceRange());
	}
}

void BloomMask::onImGui()
{
	ImGui::Checkbox("Enable", &Enable_Bloom);
	ImGui::DragFloat("Threshold", &Threshold, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	ImGui::DragFloat("Intensity", &Intensity, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.2f");
	ImGui::SliderFloat("Radius", &Radius, 0.f, 1.f, "%.2f");
}

BloomDownSample::BloomDownSample(uint32_t level) :
    m_level(level)
{
}

void BloomDownSample::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomDownSample.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	if (m_level == 1)
	{
		state.descriptor_bindings.bind(0, 0, "BloomMask", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
	else
	{
		state.descriptor_bindings.bind(0, 0, std::string("BloomLevel_") + std::to_string(m_level - 1), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	state.declareAttachment(std::string("BloomLevel_") + std::to_string(m_level), VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width >> m_level, Renderer::instance()->getRenderTargetExtent().height >> m_level);
	state.addOutputAttachment(std::string("BloomLevel_") + std::to_string(m_level), AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 1, std::string("BloomLevel_") + std::to_string(m_level), ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomDownSample::resolveResources(ResolveState &resolve)
{
}

void BloomDownSample::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (Enable_Bloom)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		auto &extent = Renderer::instance()->getRenderTargetExtent();

		vkCmdDispatch(cmd_buffer, ((extent.width >> 1) + 8 - 1) / 8, ((extent.height >> 1) + 8 - 1) / 8, 1);
	}
}

void BloomDownSample::onImGui()
{
}

BloomBlur::BloomBlur(uint32_t level) :
    m_level(level)
{
}

void BloomBlur::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomBlur.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	state.declareAttachment(std::string("BloomBlur_") + std::to_string(m_level), VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width >> m_level, Renderer::instance()->getRenderTargetExtent().height >> m_level);
	state.addOutputAttachment(std::string("BloomBlur_") + std::to_string(m_level), AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, std::string("BloomLevel_") + std::to_string(m_level), ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	state.descriptor_bindings.bind(0, 1, std::string("BloomBlur_") + std::to_string(m_level), ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomBlur::resolveResources(ResolveState &resolve)
{
}

void BloomBlur::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (Enable_Bloom)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		auto &extent = Renderer::instance()->getRenderTargetExtent();

		vkCmdDispatch(cmd_buffer, ((extent.width >> 1) + 8 - 1) / 8, ((extent.height >> 1) + 8 - 1) / 8, 1);
	}
}

void BloomBlur::onImGui()
{
}

BloomUpSample::BloomUpSample(uint32_t level, bool start) :
    m_level(level), m_start(start)
{
}

void BloomUpSample::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomUpSample.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	if (m_start)
	{
		state.descriptor_bindings.bind(0, 0, std::string("BloomBlur_") + std::to_string(m_level), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}
	else
	{
		state.descriptor_bindings.bind(0, 0, std::string("BloomLevel_") + std::to_string(m_level), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	}

	state.descriptor_bindings.bind(0, 1, std::string("BloomBlur_") + std::to_string(m_level - 1), ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 2, std::string("BloomLevel_") + std::to_string(m_level - 1), ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomUpSample::resolveResources(ResolveState &resolve)
{
}

void BloomUpSample::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (Enable_Bloom)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		auto &extent = Renderer::instance()->getRenderTargetExtent();

		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Radius), &Radius);
		vkCmdDispatch(cmd_buffer, ((extent.width >> (m_level - 1)) + 8 - 1) / 8, ((extent.height >> (m_level - 1)) + 8 - 1) / 8, 1);
	}
}

void BloomUpSample::onImGui()
{
}

BloomBlend::BloomBlend(const std::string &input, const std::string &output) :
    m_input(input), m_output(output)
{
}

void BloomBlend::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/PostProcess/BloomBlend.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	state.declareAttachment(m_output, VK_FORMAT_R16G16B16A16_SFLOAT, Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height, true);
	state.addOutputAttachment(m_output, AttachmentState::Clear_Color);

	state.descriptor_bindings.bind(0, 0, "BloomLevel_1", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, m_input, ImageViewType::Native, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	state.descriptor_bindings.bind(0, 2, m_output, ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void BloomBlend::resolveResources(ResolveState &resolve)
{
}

void BloomBlend::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	auto &extent = Renderer::instance()->getRenderTargetExtent();

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Intensity), &Intensity);
	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(Intensity), sizeof(uint32_t), reinterpret_cast<uint32_t *>(&Enable_Bloom));
	vkCmdDispatch(cmd_buffer, (extent.width + 8 - 1) / 8, (extent.height + 8 - 1) / 8, 1);
}

void BloomBlend::onImGui()
{
}
}        // namespace Ilum::pass