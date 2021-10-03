#include "Pipeline.hpp"
#include "Shader.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Graphics/Command/CommandBuffer.hpp"

namespace Ilum
{
Pipeline::Pipeline(const LogicalDevice &logical_device) :
    m_logical_device(logical_device)
{
	m_shader = createScope<Shader>();
}

Pipeline::~Pipeline()
{
	if (m_pipeline)
	{
		vkDestroyPipeline(m_logical_device, m_pipeline, nullptr);
	}

	if (m_pipeline_layout)
	{
		vkDestroyPipelineLayout(m_logical_device, m_pipeline_layout, nullptr);
	}

	if (m_descriptor_set_layout)
	{
		vkDestroyDescriptorSetLayout(m_logical_device, m_descriptor_set_layout, nullptr);
	}

	if (m_descriptor_pool)
	{
		vkDestroyDescriptorPool(m_logical_device, m_descriptor_pool, nullptr);
	}
}

void Pipeline::bind(const CommandBuffer &command_buffer)
{
	vkCmdBindPipeline(command_buffer, m_pipeline_bind_point, m_pipeline);
}

const VkDescriptorSetLayout &Pipeline::getDescriptorSetLayout() const
{
	return m_descriptor_set_layout;
}

const VkDescriptorPool &Pipeline::getDescriptorPool() const
{
	return m_descriptor_pool;
}

const VkPipeline &Pipeline::getPipeline() const
{
	return m_pipeline;
}

const VkPipelineLayout &Pipeline::getPipelineLayout() const
{
	return m_pipeline_layout;
}

const VkPipelineBindPoint &Pipeline::getPipelineBindPoint() const
{
	return m_pipeline_bind_point;
}

const Shader &Pipeline::getShader() const
{
	return *m_shader;
}
}        // namespace Ilum