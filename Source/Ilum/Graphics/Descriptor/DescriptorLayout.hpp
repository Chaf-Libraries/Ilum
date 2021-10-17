#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Shader;

class DescriptorLayout
{
  public:
	DescriptorLayout(const Shader &shader, const uint32_t set_index);

	~DescriptorLayout();

	const VkDescriptorSetLayout &getDescriptorSetLayout() const;

	operator const VkDescriptorSetLayout &() const;

	const std::vector<VkDescriptorSetLayoutBinding> &getBindings() const;

	const std::vector<VkDescriptorBindingFlags>& getBindingFlags() const;

  private:
	VkDescriptorSetLayout m_handle = VK_NULL_HANDLE;
	uint32_t              m_set_index;

	std::vector<VkDescriptorSetLayoutBinding> m_bindings;
	std::vector<VkDescriptorBindingFlags>     m_binding_flags;
};
}        // namespace Ilum