#include "Shader.hpp"
#include "../Device/Device.hpp"

#include <Core/FileSystem.hpp>

namespace Ilum::Graphics
{
Shader::Shader(const Device &device, const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage) :
    m_stage(stage),
    m_device(device)
{
	VkShaderModuleCreateInfo shader_module_info = {};
	shader_module_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_info.pCode                    = spirv.data();

	vkCreateShaderModule(device, &shader_module_info, nullptr, &m_handle);

	m_reflection_data = SpirvReflection::Reflect(spirv, stage);
}

Shader ::~Shader()
{
	if (m_handle)
	{
		vkDestroyShaderModule(m_device, m_handle, nullptr);
	}
}

Shader::operator const VkShaderModule &() const
{
	return m_handle;
}

const VkShaderModule &Shader::GetHandle() const
{
	return m_handle;
}

VkShaderStageFlagBits Shader::GetStage() const
{
	return m_stage;
}

const ReflectionData &Shader::GetReflectionData() const
{
	return m_reflection_data;
}
}