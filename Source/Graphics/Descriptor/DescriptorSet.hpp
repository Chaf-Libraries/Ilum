#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Pipeline;
class CommandBuffer;
class Shader;

class DescriptorSet
{
  public:
	DescriptorSet(const Shader &shader, uint32_t set_index = 0);

	~DescriptorSet();

	void update(const std::vector<VkWriteDescriptorSet> &write_descriptor_sets) const;

	const VkDescriptorSet &getDescriptorSet() const;

	operator const VkDescriptorSet &() const;

	uint32_t index() const;

  private:
	VkDescriptorSet     m_handle = VK_NULL_HANDLE;
	uint32_t            m_set_index = 0;
};
}        // namespace Ilum