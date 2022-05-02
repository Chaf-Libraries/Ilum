#pragma once

#include "ShaderReflection.hpp"

#include <unordered_map>

namespace Ilum
{
class RHIDevice;

class DescriptorLayout
{
  public:
	DescriptorLayout(RHIDevice *device, const ShaderReflectionData &meta, const uint32_t set_index);
	~DescriptorLayout();

	DescriptorLayout(const DescriptorLayout &) = delete;
	DescriptorLayout &operator=(const DescriptorLayout &) = delete;
	DescriptorLayout(DescriptorLayout &&other) noexcept;
	DescriptorLayout &operator=(DescriptorLayout &&other) noexcept;

	operator VkDescriptorSetLayout() const;

	const VkDescriptorSetLayout &GetHandle() const;

	const std::vector<VkDescriptorSetLayoutBinding> &GetBindings() const;
	const std::vector<VkDescriptorBindingFlags>     &GetBindingFlags() const;

  private:
	RHIDevice *p_device = nullptr;

	VkDescriptorSetLayout m_handle = VK_NULL_HANDLE;
	uint32_t              m_set_index;

	std::vector<VkDescriptorSetLayoutBinding> m_bindings;
	std::vector<VkDescriptorBindingFlags>     m_binding_flags;
};

class DescriptorPool
{
  public:
	static const uint32_t MAX_SETS_PER_POOL = 16;

  public:
	DescriptorPool(RHIDevice *device, const DescriptorLayout &descriptor_layout, uint32_t pool_size = MAX_SETS_PER_POOL);
	~DescriptorPool();

	DescriptorPool(const DescriptorPool &) = delete;
	DescriptorPool &operator=(const DescriptorPool &) = delete;
	DescriptorPool(DescriptorPool &&other) noexcept;
	DescriptorPool &operator=(DescriptorPool &&other) noexcept;

	void            Reset();
	bool            Has(VkDescriptorSet descriptor_set);
	VkDescriptorSet Allocate(const DescriptorLayout &descriptor_layout);
	void            Free(VkDescriptorSet descriptor_set);

  private:
	uint32_t FindAvaliablePool(const DescriptorLayout &descriptor_layout, uint32_t pool_index);

  private:
	RHIDevice *p_device = nullptr;

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

class DescriptorAllocator
{
  public:
	DescriptorAllocator(RHIDevice *device);
	~DescriptorAllocator() = default;

	DescriptorAllocator(const DescriptorAllocator &) = delete;
	DescriptorAllocator &operator=(const DescriptorAllocator &) = delete;
	DescriptorAllocator(DescriptorAllocator &&other)            = delete;
	DescriptorAllocator &operator=(DescriptorAllocator &&other) = delete;

	VkDescriptorSetLayout GetDescriptorLayout(const ShaderReflectionData &meta, uint32_t set_index);
	VkDescriptorSet       AllocateDescriptorSet(const ShaderReflectionData &meta, uint32_t set_index);
	VkDescriptorSet       AllocateDescriptorSet(const VkDescriptorSetLayout &descriptor_layout);

	void Free(const VkDescriptorSet &descriptor_set);

  private:
	RHIDevice *p_device = nullptr;

	std::vector<DescriptorLayout> m_descriptor_layouts;

	// Shader hash - DescriptorSetLayout index
	std::unordered_map<size_t, size_t> m_hash_layout_mapping;

	// VkDescriptorSetLayout - DescriptorSetLayout index
	std::unordered_map<VkDescriptorSetLayout, size_t> m_descriptor_layout_table;

	std::vector<DescriptorPool> m_descriptor_pools;

	// VkDescriptorSetLayout - Descriptor Pool Index
	std::unordered_map<VkDescriptorSetLayout, size_t> m_descriptor_pool_table;

	// VkDescriptorSet -  Descriptor Pool Index
	std::unordered_map<VkDescriptorSet, size_t> m_set_pool_mapping;
};

}        // namespace Ilum