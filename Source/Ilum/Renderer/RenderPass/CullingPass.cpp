#pragma once

#include "CullingPass.hpp"

#include "Renderer/Renderer.hpp"

namespace Ilum::pass
{
CullingPass::CullingPass()
{
	m_buffer = createScope<Buffer>(10000000 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
}

void CullingPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/culling.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	state.descriptor_bindings.bind(0, 0, "BoundingBox", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 1, "IndirectDrawCommand", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 2, "TransformData", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 3, "Camera", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	state.descriptor_bindings.bind(0, 4, "Meshlet", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	state.descriptor_bindings.bind(0, 5, "Count", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void CullingPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("BoundingBox", Renderer::instance()->getBuffer(Renderer::BufferType::BoundingBox));
	resolve.resolve("IndirectDrawCommand", Renderer::instance()->getBuffer(Renderer::BufferType::IndirectCommand));
	resolve.resolve("TransformData", Renderer::instance()->getBuffer(Renderer::BufferType::Transform));
	resolve.resolve("Camera", Renderer::instance()->getBuffer(Renderer::BufferType::MainCamera));
	resolve.resolve("Meshlet", Renderer::instance()->getBuffer(Renderer::BufferType::Meshlet));
	resolve.resolve("Count", *m_buffer);
}

void CullingPass::render(RenderPassState &state)
{
	std::vector<uint32_t> meshlet_result(Renderer::instance()->Meshlet_Count);
	std::memcpy(meshlet_result.data(), m_buffer->map(), Renderer::instance()->Meshlet_Count * sizeof(uint32_t));
	m_buffer->unmap();
	uint32_t visible = 0;
	for (uint32_t i : meshlet_result)
	{
		visible += i;
	}

	//LOG_INFO("{}/{}", visible, Renderer::instance()->Meshlet_Count);

	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdDispatch(cmd_buffer, (Renderer::instance()->Meshlet_Count + 64 - 1) / 64, 1, 1);
}
}        // namespace Ilum::pass