#include "DescriptorCache.hpp"
#include "../Device/Device.hpp"
#include "../Shader/SpirvReflection.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSetLayout.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Graphics
{
DescriptorCache::DescriptorCache(const Device &device) :
    m_device(device)
{
}

const DescriptorSetLayout &DescriptorCache::RequestDescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set)
{
	size_t hash = 0;
	Core::HashCombine(hash, reflection_data.hash);
	Core::HashCombine(hash, set);

	if (m_descriptor_layouts.find(hash) != m_descriptor_layouts.end())
	{
		return *m_descriptor_layouts[hash];
	}

	{
		std::lock_guard<std::mutex> lock(m_layout_mutex);
		m_descriptor_layouts.emplace(hash, std::make_unique<DescriptorSetLayout>(m_device, reflection_data, set));
	}

	return *m_descriptor_layouts[hash];
}

VkDescriptorSet DescriptorCache::AllocateDescriptorSet(const ReflectionData &reflection_data, uint32_t set)
{
	std::lock_guard<std::mutex> lock(m_set_mutex);

	auto &pool           = RequestDescriptorPool(reflection_data, set);
	auto  descriptor_set = pool.Allocate();

	return descriptor_set;
}

void DescriptorCache::Free(const VkDescriptorSet &descriptor_set)
{
	std::lock_guard<std::mutex> lock(m_set_mutex);

	for (auto &[hash, pool] : m_descriptor_pools)
	{
		if (pool->Has(descriptor_set))
		{
			pool->Free(descriptor_set);
			return;
		}
	}
}

DescriptorPool &DescriptorCache::RequestDescriptorPool(const ReflectionData &reflection_data, uint32_t set)
{
	size_t hash = 0;
	Core::HashCombine(hash, reflection_data.hash);
	Core::HashCombine(hash, set);

	if (m_descriptor_pools.find(hash) != m_descriptor_pools.end())
	{
		return *m_descriptor_pools[hash];
	}

	{
		std::lock_guard<std::mutex> lock(m_pool_mutex);
		m_descriptor_pools.emplace(hash, std::make_unique<DescriptorPool>(m_device, RequestDescriptorSetLayout(reflection_data, set)));
	}

	return *m_descriptor_pools[hash];
}
}        // namespace Ilum::Graphics