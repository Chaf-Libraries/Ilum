#include "Test.hpp"
#include <glm/glm.hpp>

#include <Graphics/Vulkan/VK_Debugger.h>

#include <imgui.h>

namespace Ilum::pass
{
TestPass::TestPass()
{
	std::vector<glm::vec3> data1 = {
	    glm::vec3(1.f, 0.f, 0.f),
	    glm::vec3(0.f, 1.f, 0.f),
	    glm::vec3(0.f, 0.f, 1.f)};

	std::vector<glm::vec3> data2 = {
	    glm::vec3(0.5f, 0.f, 0.f),
	    glm::vec3(0.f, 0.5f, 0.f),
	    glm::vec3(0.f, 0.f, 0.5f)};

	std::vector<glm::vec3> data3 = {
	    glm::vec3(0.1f, 0.f, 0.f),
	    glm::vec3(0.f, 0.1f, 0.f),
	    glm::vec3(0.f, 0.f, 0.1f)};

	buffer1 = Buffer(data1.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	buffer2 = Buffer(data2.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
	buffer3 = Buffer(data3.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

	{
		CommandBuffer cmd_buffer(QueueUsage::Compute);
		cmd_buffer.begin();

		Buffer staging1 = Buffer(data3.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		std::memcpy(staging1.map(), data1.data(), data1.size() * sizeof(glm::vec3));
		cmd_buffer.copyBuffer(BufferInfo{staging1}, BufferInfo{buffer1}, data1.size() * sizeof(glm::vec3));

		Buffer staging2 = Buffer(data3.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		std::memcpy(staging2.map(), data2.data(), data2.size() * sizeof(glm::vec3));
		cmd_buffer.copyBuffer(BufferInfo{staging2}, BufferInfo{buffer2}, data2.size() * sizeof(glm::vec3));

		Buffer staging3 = Buffer(data3.size() * sizeof(glm::vec3), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
		std::memcpy(staging3.map(), data3.data(), data3.size() * sizeof(glm::vec3));
		cmd_buffer.copyBuffer(BufferInfo{staging3}, BufferInfo{buffer3}, data3.size() * sizeof(glm::vec3));

		cmd_buffer.end();
		cmd_buffer.submitIdle();
	}

	address_buffer    = Buffer(3 * sizeof(uint64_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	uint64_t *address = (uint64_t *) address_buffer.map();
	address[0]        = buffer1.getDeviceAddress();
	address[1]        = buffer2.getDeviceAddress();
	address[2]        = buffer3.getDeviceAddress();
	address_buffer.unmap();

	VK_Debugger::setName(buffer1, "buffer1");
	VK_Debugger::setName(buffer2, "buffer2");
	VK_Debugger::setName(buffer3, "buffer3");
	VK_Debugger::setName(address_buffer, "address_buffer");

	m_push_constant.buffer_address = buffer1.getDeviceAddress();
	m_push_constant.color_index    = 0;
}

void TestPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Source/Shaders/Test.hlsl", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::HLSL, "main");

	state.descriptor_bindings.bind(0, 0, "Result", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	//state.descriptor_bindings.bind(0, 1, "Address", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

	state.declareAttachment("Result", VK_FORMAT_R16G16B16A16_SFLOAT, 1024, 1024);
	state.addOutputAttachment("Result", AttachmentState::Clear_Color);
}

void TestPass::resolveResources(ResolveState &resolve)
{
	//resolve.resolve("Address", address_buffer);
}

void TestPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (auto &descriptor_set : state.pass.descriptor_sets)
	{
		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, descriptor_set.index(), 1, &descriptor_set.getDescriptorSet(), 0, nullptr);
	}

	vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_push_constant), &m_push_constant);
	vkCmdDispatch(cmd_buffer, 1024 / 32, 1024 / 32, 1);
}

void TestPass::onImGui()
{
	if (ImGui::SliderInt("Buffer index", &buffer_index, 0, 2))
	{
		if (buffer_index == 0)
			m_push_constant.buffer_address = buffer1.getDeviceAddress();
		if (buffer_index == 1)
			m_push_constant.buffer_address = buffer2.getDeviceAddress();
		if (buffer_index == 2)
			m_push_constant.buffer_address = buffer3.getDeviceAddress();
	}
	ImGui::SliderInt("Color index", reinterpret_cast<int *>(&m_push_constant.color_index), 0, 2);
	ImGui::ShowDemoWindow();
}

}        // namespace Ilum::pass