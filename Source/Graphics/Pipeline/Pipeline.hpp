#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;
class LogicalDevice;
class Shader;
class DescriptorLayout;

class Pipeline
{
  public:
	Pipeline();

	~Pipeline();

	void bind(const CommandBuffer &command_buffer);

	const VkPipeline &getPipeline() const;

	const VkPipelineLayout &getPipelineLayout() const;

	const VkPipelineBindPoint &getPipelineBindPoint() const;

	const Shader &getShader() const;

  protected:
	void createPipelineLayout();

  protected:
	scope<Shader> m_shader = nullptr;

	VkPipeline          m_pipeline            = VK_NULL_HANDLE;
	VkPipelineLayout    m_pipeline_layout     = VK_NULL_HANDLE;
	VkPipelineBindPoint m_pipeline_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

	std::map<VkShaderStageFlagBits, VkSpecializationInfo> m_specialization_infos;
};
}        // namespace Ilum