#include "RayTracingTestPass.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

#include "ImGui/ImGuiContext.hpp"

#include "Renderer/Renderer.hpp"

#include <imgui.h>

namespace Ilum::pass
{
RayTracingTestPass::RayTracingTestPass()
{
	m_render_result = Image(Renderer::instance()->getRenderTargetExtent().width, Renderer::instance()->getRenderTargetExtent().height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	VK_Debugger::setName(m_render_result, "ray tracing test result");

	{
		CommandBuffer cmd_buffer;
		cmd_buffer.begin();
		cmd_buffer.transferLayout(m_render_result, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, VK_IMAGE_USAGE_SAMPLED_BIT);
		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}
}

void RayTracingTestPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/RTX/raygen.rgen", VK_SHADER_STAGE_RAYGEN_BIT_KHR, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/RTX/miss.rmiss", VK_SHADER_STAGE_MISS_BIT_KHR, Shader::Type::GLSL);
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/RTX/closesthit.rchit", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "TLAS", VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	state.descriptor_bindings.bind(0, 1, "Render Result", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	state.descriptor_bindings.bind(0, 2, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

void RayTracingTestPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("TLAS", Renderer::instance()->Render_Buffer.Top_Level_AS);
	resolve.resolve("Render Result", m_render_result);
	resolve.resolve("Camera", Renderer::instance()->Render_Buffer.Camera_Buffer);
}

void RayTracingTestPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	if (Renderer::instance()->Render_Stats.static_mesh_count.meshlet_count > 0)
	{
		vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

		for (auto &descriptor_set : state.pass.descriptor_sets)
		{
			vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
		}

		vkCmdTraceRaysKHR(
		    cmd_buffer,
		    state.pass.shader_binding_table.raygen->getHandle(),
		    state.pass.shader_binding_table.miss->getHandle(),
		    state.pass.shader_binding_table.hit->getHandle(),
		    state.pass.shader_binding_table.callable->getHandle(),
		    Renderer::instance()->getRenderTargetExtent().width,
		    Renderer::instance()->getRenderTargetExtent().height,
		    1);
	}
}

void RayTracingTestPass::onImGui()
{
	if (m_render_result)
	{
		ImGui::Text("Ray Tracing Result:");
		ImGui::Image(ImGuiContext::textureID(m_render_result.getView(), Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)), ImVec2(400, 400));
	}
}
}        // namespace Ilum::pass