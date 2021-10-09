#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class DescriptorLayout;
class DescriptorSet;

class DescriptorPool
{
  public:
	static const uint32_t MAX_SETS_PER_POOL = 16;

  public:
	DescriptorPool(const DescriptorLayout &descriptor_layout, uint32_t pool_size = MAX_SETS_PER_POOL);

	~DescriptorPool();

	void reset();

	bool has(VkDescriptorSet descriptor_set);

	VkDescriptorSet allocate(const DescriptorLayout &descriptor_layout);

	void free(VkDescriptorSet descriptor_set);

  private:
	uint32_t find_avaliable_pool(const DescriptorLayout &descriptor_layout, uint32_t pool_index);

  private:
	// Current pool index to allocate descriptor set
	uint32_t m_pool_index = 0;

	// Max number of sets for each pool
	uint32_t m_pool_max_sets = 0;

	std::vector<VkDescriptorPoolSize> m_pool_sizes;

	std::vector<VkDescriptorPool> m_descriptor_pools;

	// Count sets for each pool
	std::vector<uint32_t> m_pool_sets_count;

	// Descriptor set - pool index
	std::unordered_map<VkDescriptorSet, uint32_t> m_set_pool_mapping;
};
}        // namespace Ilum