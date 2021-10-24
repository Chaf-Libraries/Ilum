#include "ShaderCache.hpp"
#include "ShaderCompiler.hpp"

#include "File/FileSystem.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Device/LogicalDevice.hpp"

namespace Ilum
{
ShaderCache::~ShaderCache()
{
	for (auto &shader_module : m_shader_modules)
	{
		if (shader_module != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(GraphicsContext::instance()->getLogicalDevice(), shader_module, nullptr);
		}
	}

	m_shader_modules.clear();
}

VkShaderModule ShaderCache::load(const std::string &filename, VkShaderStageFlagBits stage, Shader::Type type)
{
	// Look for shader module
	if (m_lookup.find(filename) != m_lookup.end())
	{
		return m_shader_modules.at(m_lookup[filename]);
	}

	std::vector<uint8_t> raw_data;
	FileSystem::read(filename, raw_data, type == Shader::Type::SPIRV);

	auto spirv = ShaderCompiler::compile(raw_data, stage, type);
	m_reflection_data.emplace_back(std::move(ShaderReflection::reflect(spirv, stage)));

	VkShaderModuleCreateInfo shader_module_create_info = {};
	shader_module_create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_create_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_create_info.pCode                    = spirv.data();

	VkShaderModule shader_module;
	if (!VK_CHECK(vkCreateShaderModule(GraphicsContext::instance()->getLogicalDevice(), &shader_module_create_info, nullptr, &shader_module)))
	{
		VK_ERROR("Failed to create shader module");
		return VK_NULL_HANDLE;
	}

	m_shader_modules.push_back(shader_module);
	m_lookup[filename] = m_shader_modules.size() - 1;
	m_mapping[shader_module] = m_shader_modules.size() - 1;

	return shader_module;
}

const ReflectionData &ShaderCache::reflect(VkShaderModule shader)
{
	ASSERT(m_mapping.find(shader) != m_mapping.end());
	return m_reflection_data.at(m_mapping.at(shader));
}
}        // namespace Ilum