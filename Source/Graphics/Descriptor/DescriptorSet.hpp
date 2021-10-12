#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class Pipeline;
class CommandBuffer;

class DescriptorSet
{
  public:
	DescriptorSet(const Pipeline *pipeline, uint32_t set_index = 0);

	~DescriptorSet();

	void update(const std::vector<VkWriteDescriptorSet> &write_descriptor_sets);

	const VkDescriptorSet &getDescriptorSet() const;

	operator const VkDescriptorSet &() const;

	void bind(const CommandBuffer &command_buffer);

  private:
	VkDescriptorSet     m_handle = VK_NULL_HANDLE;
	VkPipelineLayout    m_pipeline_layout;
	VkPipelineBindPoint m_pipeline_bind_point;
	uint32_t            m_set_index = 0;
};
}        // namespace Ilum