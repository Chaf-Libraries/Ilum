#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
struct ReflectionData;

class DescriptorSetLayout
{
  public:
	DescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set);
	~DescriptorSetLayout();

	DescriptorSetLayout(const DescriptorSetLayout &) = delete;
	DescriptorSetLayout &operator=(const DescriptorSetLayout &) = delete;
	DescriptorSetLayout(DescriptorSetLayout &&other)            = delete;
	DescriptorSetLayout &operator=(DescriptorSetLayout &&other) = delete;

	operator const VkDescriptorSetLayout &() const;

	const VkDescriptorSetLayout &GetHandle() const;

	uint32_t GetSet() const;

	const std::vector<VkDescriptorSetLayoutBinding> &GetBinding() const;

  private:
	VkDescriptorSetLayout                     m_handle = VK_NULL_HANDLE;
	std::vector<VkDescriptorSetLayoutBinding> m_bindings;
	uint32_t                                  m_set = 0;
};

class DescriptorPool
{
	static const uint32_t MAX_SETS_PER_POOL = 16;

  public:
	DescriptorPool(const DescriptorSetLayout &descriptor_layout, uint32_t pool_size = MAX_SETS_PER_POOL);
	~DescriptorPool();

	DescriptorPool(const DescriptorPool &) = delete;
	DescriptorPool &operator=(const DescriptorPool &) = delete;
	DescriptorPool(DescriptorPool &&other)            = delete;
	DescriptorPool &operator=(DescriptorPool &&other) = delete;

	void            Reset();
	bool            Has(VkDescriptorSet descriptor_set);
	VkDescriptorSet Allocate();
	void            Free(const VkDescriptorSet &descriptor_set);

  private:
	uint32_t FindAvaliablePool(uint32_t pool_index);

  private:
	const DescriptorSetLayout &m_descriptor_layout;

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

class DescriptorCache
{
  public:
	DescriptorCache()  = default;
	~DescriptorCache() = default;

	const DescriptorSetLayout &RequestDescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name = "");
	VkDescriptorSet            RequestDescriptorSet(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name = "");

	// Not recommand
	void Free(const VkDescriptorSet &descriptor_set);

  private:
	DescriptorPool &RequestDescriptorPool(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name = "");

  private:
	std::unordered_map<size_t, std::unique_ptr<DescriptorSetLayout>> m_descriptor_layouts;
	std::unordered_map<size_t, std::unique_ptr<DescriptorPool>>      m_descriptor_pools;
};
}        // namespace Ilum::Vulkan