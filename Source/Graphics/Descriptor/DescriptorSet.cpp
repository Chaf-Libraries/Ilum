#include "DescriptorSet.hpp"
#include "DescriptorCache.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/Pipeline.hpp"

namespace Ilum
{
DescriptorSet::DescriptorSet(const Pipeline *pipeline, uint32_t set_index) :
    m_pipeline_layout(pipeline->getPipelineLayout()),
    m_pipeline_bind_point(pipeline->getPipelineBindPoint()),
    m_set_index(set_index)
{
	m_handle = GraphicsContext::instance()->getDescriptorCache().allocateDescriptorSet(pipeline, set_index);
}

DescriptorSet::~DescriptorSet()
{
	GraphicsContext::instance()->getDescriptorCache().free(m_handle);
}

void DescriptorSet::update(const std::vector<VkWriteDescriptorSet> &write_descriptor_sets)
{
	vkUpdateDescriptorSets(GraphicsContext::instance()->getLogicalDevice(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
}

const VkDescriptorSet &DescriptorSet::getDescriptorSet() const
{
	return m_handle;
}

DescriptorSet::operator const VkDescriptorSet &() const
{
	return m_handle;
}

void DescriptorSet::bind(const CommandBuffer &command_buffer)
{
	vkCmdBindDescriptorSets(command_buffer, m_pipeline_bind_point, m_pipeline_layout, m_set_index, 1, &m_handle, 0, nullptr);
}
}        // namespace Ilum