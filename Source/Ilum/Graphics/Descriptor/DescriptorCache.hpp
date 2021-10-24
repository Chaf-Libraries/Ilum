#pragma once

#include "Utils/PCH.hpp"

#include "DescriptorLayout.hpp"
#include "DescriptorPool.hpp"

namespace Ilum
{
class Shader;

class DescriptorCache
{
  public:
	DescriptorCache() = default;

	~DescriptorCache() = default;

	VkDescriptorSetLayout getDescriptorLayout(const Shader &shader, uint32_t set_index);

	VkDescriptorSet allocateDescriptorSet(const Shader &shader, uint32_t set_index);

	void free(const VkDescriptorSet &descriptor_set);

  private:
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