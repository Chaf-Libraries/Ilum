#pragma once

#include "../Vulkan.hpp"

namespace Ilum::Graphics
{
struct ReflectionData;
class DescriptorSetLayout;
class DescriptorPool;
class Device;

class DescriptorCache
{
  public:
	DescriptorCache(const Device& device);
	~DescriptorCache() = default;

	const DescriptorSetLayout &RequestDescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set);
	VkDescriptorSet            AllocateDescriptorSet(const ReflectionData &reflection_data, uint32_t set);

	// Not recommand
	void Free(const VkDescriptorSet &descriptor_set);

  private:
	DescriptorPool &RequestDescriptorPool(const ReflectionData &reflection_data, uint32_t set);

  private:
	const Device &m_device;

	std::mutex m_pool_mutex;
	std::mutex m_layout_mutex;
	std::mutex m_set_mutex;

	std::unordered_map<size_t, std::unique_ptr<DescriptorSetLayout>> m_descriptor_layouts;
	std::unordered_map<size_t, std::unique_ptr<DescriptorPool>>      m_descriptor_pools;
};
}