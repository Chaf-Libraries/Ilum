#pragma once

#include "Graphics/Vulkan.hpp"

namespace Ilum::Graphics
{
class Device;
struct ReflectionData;

class PipelineLayout
{
  public:
	PipelineLayout(const Device &device, const ReflectionData &reflection_data, const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts);
	~PipelineLayout();

	PipelineLayout(const PipelineLayout &) = delete;
	PipelineLayout &operator=(const PipelineLayout &) = delete;
	PipelineLayout(PipelineLayout &&)                 = delete;
	PipelineLayout &operator=(PipelineLayout &&) = delete;

	operator const VkPipelineLayout &() const;

	const VkPipelineLayout &GetHandle() const;

  private:
	const Device &   m_device;
	VkPipelineLayout m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Graphics