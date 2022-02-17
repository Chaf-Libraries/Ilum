#include "DescriptorPool.hpp"
#include "DescriptorSetLayout.hpp"
#include "../Device/Device.hpp"

#include <map>

namespace Ilum::Graphics
{
DescriptorPool::DescriptorPool(const Device &device, const DescriptorSetLayout &descriptor_layout, uint32_t pool_size) :
    m_device(device), m_pool_max_sets(pool_size), m_descriptor_layout(descriptor_layout)
{
	auto &bindings = descriptor_layout.GetBinding();

	std::map<VkDescriptorType, uint32_t> descriptor_type_counts;

	for (auto &binding : bindings)
	{
		descriptor_type_counts[binding.descriptorType] += binding.descriptorCount;
	}

	m_pool_sizes.resize(descriptor_type_counts.size());

	auto pool_size_it = m_pool_sizes.begin();

	for (auto &it : descriptor_type_counts)
	{
		pool_size_it->type            = it.first;
		pool_size_it->descriptorCount = it.second * pool_size;
		++pool_size_it;
	}
}

DescriptorPool::~DescriptorPool()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkDestroyDescriptorPool(m_device, pool, nullptr);
	}
}

VkDescriptorSet DescriptorPool::Allocate()
{
	m_pool_index = FindAvaliablePool(m_pool_index);

	++m_pool_sets_count[m_pool_index];

	VkDescriptorSetAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.descriptorPool              = m_descriptor_pools[m_pool_index];
	allocate_info.descriptorSetCount          = 1;
	allocate_info.pSetLayouts                 = &m_descriptor_layout.GetHandle();

	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

	if (!VK_CHECK(vkAllocateDescriptorSets(m_device, &allocate_info, &descriptor_set)))
	{
		LOG_ERROR("Failed to allocate descriptor set!");

		--m_pool_sets_count[m_pool_index];
		return VK_NULL_HANDLE;
	}

	m_set_pool_mapping.emplace(descriptor_set, m_pool_index);

	return descriptor_set;
}

void DescriptorPool::Reset()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkResetDescriptorPool(m_device, pool, 0);
	}

	std::fill(m_pool_sets_count.begin(), m_pool_sets_count.end(), 0);
	m_set_pool_mapping.clear();

	m_pool_index = 0;
}

bool DescriptorPool::Has(VkDescriptorSet descriptor_set)
{
	return m_set_pool_mapping.find(descriptor_set) != m_set_pool_mapping.end();
}

void DescriptorPool::Free(const VkDescriptorSet &descriptor_set)
{
	auto it = m_set_pool_mapping.find(descriptor_set);

	if (it == m_set_pool_mapping.end())
	{
		return;
	}

	auto desc_pool_index = it->second;

	vkFreeDescriptorSets(m_device, m_descriptor_pools[desc_pool_index], 1, &descriptor_set);

	m_set_pool_mapping.erase(it);

	--m_pool_sets_count[desc_pool_index];

	m_pool_index = desc_pool_index;
}

uint32_t DescriptorPool::FindAvaliablePool(uint32_t pool_index)
{
	// Create a new pool
	if (m_descriptor_pools.size() <= pool_index)
	{
		VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
		descriptor_pool_create_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptor_pool_create_info.pPoolSizes                 = m_pool_sizes.data();
		descriptor_pool_create_info.poolSizeCount              = static_cast<uint32_t>(m_pool_sizes.size());
		descriptor_pool_create_info.maxSets                    = m_pool_max_sets;

		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

		if (!VK_CHECK(vkCreateDescriptorPool(m_device, &descriptor_pool_create_info, nullptr, &descriptor_pool)))
		{
			LOG_ERROR("Failed to create descriptor pool!");
			return 0;
		}

		m_descriptor_pools.push_back(descriptor_pool);
		m_pool_sets_count.push_back(0);
		return pool_index;
	}
	else if (m_pool_sets_count[pool_index] < m_pool_max_sets)
	{
		return pool_index;
	}

	return FindAvaliablePool(++pool_index);
}
}        // namespace Ilum::Graphics