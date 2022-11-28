#pragma once

#include "Fwd.hpp"

namespace Ilum::Vulkan
{
class Shader : public RHIShader
{
  public:
	Shader(RHIDevice *device, const std::string& entry_point, const std::vector<uint8_t> &source);
	
	virtual ~Shader() override;

	VkShaderModule GetHandle() const;

  private:
	VkShaderModule m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan