#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
class DescriptorSetLayout;
class Device;

class DescriptorPool
{
	static const uint32_t MAX_SETS_PER_POOL = 16;

  public:
	DescriptorPool(const Device& device, const DescriptorSetLayout &descriptor_layout, uint32_t pool_size = MAX_SETS_PER_POOL);
	~DescriptorPool();

	DescriptorPool(const DescriptorPool &) = delete;
	DescriptorPool &operator=(const DescriptorPool &) = delete;
	DescriptorPool(DescriptorPool &&other)            = delete;
	DescriptorPool &operator=(DescriptorPool &&other) = delete;

	VkDescriptorSet Allocate();

	void Reset();
	bool Has(VkDescriptorSet descriptor_set);
	void Free(const VkDescriptorSet &descriptor_set);

  private:
	uint32_t FindAvaliablePool(uint32_t pool_index);

  private:
	const Device &             m_device;
	const DescriptorSetLayout &m_descriptor_layout;

	// Current pool index to allocate descriptor set
	uint32_t m_pool_index = 0;

	// Max number of sets for each pool
	uint32_t m_pool_max_sets = 0;

	std::vector<VkDescriptorPoolSize> m_pool_sizes;
	std::vector<VkDescriptorPool>     m_descriptor_pools;

	// Count sets for each pool
	std::vector<uint32_t> m_pool_sets_count;

	// Descriptor set - pool index
	std::unordered_map<VkDescriptorSet, uint32_t> m_set_pool_mapping;
};
}        // namespace Ilum::Graphics