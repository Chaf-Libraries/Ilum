#include "Pipeline.hpp"
#include "Shader.hpp"

#include "Core/Device/LogicalDevice.hpp"
#include "Core/Graphics/Command/CommandBuffer.hpp"
#include "Core/Graphics/GraphicsContext.hpp"
#include "Core/Graphics/Descriptor/DescriptorLayout.hpp"
#include "Core/Graphics/Descriptor/DescriptorCache.hpp"

namespace Ilum
{
Pipeline::Pipeline()
{
	m_shader = createScope<Shader>();
}

Pipeline::~Pipeline()
{
	if (m_pipeline)
	{
		vkDestroyPipeline(GraphicsContext::instance()->getLogicalDevice(), m_pipeline, nullptr);
	}

	if (m_pipeline_layout)
	{
		vkDestroyPipelineLayout(GraphicsContext::instance()->getLogicalDevice(), m_pipeline_layout, nullptr);
	}
}

void Pipeline::bind(const CommandBuffer &command_buffer)
{
	vkCmdBindPipeline(command_buffer, m_pipeline_bind_point, m_pipeline);
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

void Pipeline::createPipelineLayout()
{
	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	for (auto &set_index : m_shader->getSets())
	{
		descriptor_set_layouts.push_back(GraphicsContext::instance()->getDescriptorCache().getDescriptorLayout(this, set_index));
	}

	auto push_constant_ranges = m_shader->getPushConstantRanges();

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(descriptor_set_layouts.size());
	pipeline_layout_create_info.pSetLayouts                = descriptor_set_layouts.data();
	pipeline_layout_create_info.pushConstantRangeCount     = static_cast<uint32_t>(push_constant_ranges.size());
	pipeline_layout_create_info.pPushConstantRanges        = push_constant_ranges.data();

	vkCreatePipelineLayout(GraphicsContext::instance()->getLogicalDevice(), &pipeline_layout_create_info, nullptr, &m_pipeline_layout);
}
}        // namespace Ilum