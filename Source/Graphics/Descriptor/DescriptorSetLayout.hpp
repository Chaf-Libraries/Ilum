#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
struct ReflectionData;
class Device;

class DescriptorSetLayout
{
  public:
	DescriptorSetLayout(const Device &device, const ReflectionData &reflection_data, uint32_t set);
	~DescriptorSetLayout();

	DescriptorSetLayout(const DescriptorSetLayout &) = delete;
	DescriptorSetLayout &operator=(const DescriptorSetLayout &) = delete;
	DescriptorSetLayout(DescriptorSetLayout &&other)            = delete;
	DescriptorSetLayout &operator=(DescriptorSetLayout &&other) = delete;

	operator const VkDescriptorSetLayout &() const;

	const VkDescriptorSetLayout &                    GetHandle() const;
	uint32_t                                         GetSet() const;
	const std::vector<VkDescriptorSetLayoutBinding> &GetBinding() const;

  private:
	const Device &                            m_device;
	VkDescriptorSetLayout                     m_handle = VK_NULL_HANDLE;
	std::vector<VkDescriptorSetLayoutBinding> m_bindings;
	uint32_t                                  m_set = 0;
};
}        // namespace Ilum::Graphics