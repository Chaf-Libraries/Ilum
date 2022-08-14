#pragma once

#include "Shader.hpp"
#include "Device.hpp"

namespace Ilum::Vulkan
{
Shader::Shader(RHIDevice *device, const std::string &entry_point, const std::vector<uint8_t> &source) :
    RHIShader(device, entry_point, source)
{
	VkShaderModuleCreateInfo create_info = {};
	create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	create_info.codeSize                 = source.size();
	create_info.pCode                    = reinterpret_cast<const uint32_t *>(source.data());
	vkCreateShaderModule(static_cast<Device *>(p_device)->GetDevice(), &create_info, nullptr, &m_handle);
}

Shader::~Shader()
{
	if (m_handle)
	{
		vkDestroyShaderModule(static_cast<Device *>(p_device)->GetDevice(), m_handle, nullptr);
	}
}

VkShaderModule Shader::GetHandle()
{
	return m_handle;
}
}        // namespace Ilum::Vulkan