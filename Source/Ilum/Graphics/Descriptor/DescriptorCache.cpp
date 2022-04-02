#include "DescriptorCache.hpp"
#include "DescriptorLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSet.hpp"

#include "Graphics/Shader/Shader.hpp"

namespace Ilum
{
VkDescriptorSetLayout DescriptorCache::getDescriptorLayout(const Shader &shader, uint32_t set_index)
{
	size_t hash = 0;
	for (auto &[stage, shader_modules] : shader.getShaders())
	{
		for (auto& shader_module : shader_modules)
		{
			hash_combine(hash, shader_module.first);
		}
	}
	hash_combine(hash, set_index);

	if (m_hash_layout_mapping.find(hash) != m_hash_layout_mapping.end())
	{
		return m_descriptor_layouts[m_hash_layout_mapping[hash]];
	}

	m_descriptor_layouts.emplace_back(shader, set_index);

	m_hash_layout_mapping.emplace(hash, m_descriptor_layouts.size() - 1);
	m_descriptor_layout_table.emplace(m_descriptor_layouts.back().getDescriptorSetLayout(), m_descriptor_layouts.size() - 1);

	return m_descriptor_layouts[m_hash_layout_mapping[hash]];
}

VkDescriptorSet DescriptorCache::allocateDescriptorSet(const Shader &shader, uint32_t set_index)
{
	return allocateDescriptorSet(getDescriptorLayout(shader, set_index));
}

VkDescriptorSet DescriptorCache::allocateDescriptorSet(const VkDescriptorSetLayout &descriptor_layout)
{
	if (m_descriptor_pool_table.find(descriptor_layout) == m_descriptor_pool_table.end())
	{
		// Create new descriptor pool
		m_descriptor_pools.emplace_back(m_descriptor_layouts[m_descriptor_pool_table[descriptor_layout]]);
		m_descriptor_pool_table[descriptor_layout] = m_descriptor_pools.size() - 1;
	}

	return m_descriptor_pools[m_descriptor_pool_table[descriptor_layout]].allocate(m_descriptor_layouts[m_descriptor_pool_table[descriptor_layout]]);
}

void DescriptorCache::free(const VkDescriptorSet &descriptor_set)
{
	if (m_set_pool_mapping.find(descriptor_set) == m_set_pool_mapping.end())
	{
		return;
	}

	m_descriptor_pools[m_set_pool_mapping[descriptor_set]].free(descriptor_set);
}
}        // namespace Ilum