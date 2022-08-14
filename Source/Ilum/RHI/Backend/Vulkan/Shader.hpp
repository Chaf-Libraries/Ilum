#pragma once

#include "RHI/RHIShader.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Shader : public RHIShader
{
  public:
	Shader(RHIDevice *device, const std::string& entry_point, const std::vector<uint8_t> &source);
	
	virtual ~Shader() override;

	VkShaderModule GetHandle();

  private:
	VkShaderModule m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan