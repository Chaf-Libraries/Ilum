#pragma once

#include "Core/Engine/PCH.hpp"

namespace Ilum
{
class CommandBuffer;
class LogicalDevice;
class Shader;

class Pipeline
{
  public:
	Pipeline(const LogicalDevice &logical_device);

	~Pipeline();

	void bind(const CommandBuffer &command_buffer);

	const VkDescriptorSetLayout &getDescriptorSetLayout() const;

	const VkDescriptorPool &getDescriptorPool() const;

	const VkPipeline &getPipeline() const;

	const VkPipelineLayout &getPipelineLayout() const;

	const VkPipelineBindPoint &getPipelineBindPoint() const;

	const Shader &getShader() const;

  protected:
	const LogicalDevice &m_logical_device;

	scope<Shader> m_shader = nullptr;

	VkDescriptorSetLayout m_descriptor_set_layout = VK_NULL_HANDLE;
	VkDescriptorPool      m_descriptor_pool       = VK_NULL_HANDLE;
	VkPipeline            m_pipeline              = VK_NULL_HANDLE;
	VkPipelineLayout      m_pipeline_layout       = VK_NULL_HANDLE;
	VkPipelineBindPoint   m_pipeline_bind_point   = VK_PIPELINE_BIND_POINT_GRAPHICS;
};
}        // namespace Ilum