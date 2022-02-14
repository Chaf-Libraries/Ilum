#include "DescriptorSet.hpp"
#include "DescriptorCache.hpp"

#include <Graphics/Device/Device.hpp>
#include <Graphics/RenderContext.hpp>

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Shader/Shader.hpp"

namespace Ilum
{
DescriptorSet::DescriptorSet(const Shader &shader, uint32_t set_index) :
    m_set_index(set_index),
    m_handle(GraphicsContext::instance()->getDescriptorCache().allocateDescriptorSet(shader, set_index))
{

}

DescriptorSet::~DescriptorSet()
{
	GraphicsContext::instance()->getDescriptorCache().free(m_handle);
}

void DescriptorSet::update(const std::vector<VkWriteDescriptorSet> &write_descriptor_sets) const
{
	vkUpdateDescriptorSets(Graphics::RenderContext::GetDevice(), static_cast<uint32_t>(write_descriptor_sets.size()), write_descriptor_sets.data(), 0, nullptr);
}

const VkDescriptorSet &DescriptorSet::getDescriptorSet() const
{
	return m_handle;
}

DescriptorSet::operator const VkDescriptorSet &() const
{
	return m_handle;
}

uint32_t DescriptorSet::index() const
{
	return m_set_index;
}
}        // namespace Ilum