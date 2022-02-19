#pragma once

#include "Graphics/Vulkan.hpp"
#include "SpirvReflection.hpp"

#include <vector>

namespace Ilum::Graphics
{
class Device;

class Shader
{
  public:
	Shader(const Device& device, const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage);
	~Shader();

	Shader(const Shader &) = delete;
	Shader &operator=(const Shader &) = delete;
	Shader(Shader &&)                 = delete;
	Shader &operator=(Shader &&) = delete;

	operator const VkShaderModule &() const;

	const VkShaderModule &GetHandle() const;

	VkShaderStageFlagBits GetStage() const;

	const ReflectionData &GetReflectionData() const;

  private:
	const Device &        m_device;
	VkShaderModule        m_handle = VK_NULL_HANDLE;
	VkShaderStageFlagBits m_stage;
	ReflectionData        m_reflection_data;
};
}        // namespace Ilum::Graphics