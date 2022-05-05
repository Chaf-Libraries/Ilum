#include "ShaderAllocator.hpp"
#include "Device.hpp"

#include <Core/Macro.hpp>
#include <Core/Path.hpp>

namespace Ilum
{
ShaderAllocator::ShaderAllocator(RHIDevice *device) :
    p_device(device)
{
}

ShaderAllocator::~ShaderAllocator()
{
	for (auto &shader : m_shader_modules)
	{
		vkDestroyShaderModule(p_device->GetDevice(), shader, nullptr);
	}
	m_shader_modules.clear();
}

VkShaderModule ShaderAllocator::Load(const ShaderDesc &desc)
{
	size_t shader_hash = desc.Hash();
	if (m_lookup.find(shader_hash) != m_lookup.end())
	{
		return m_shader_modules.at(m_lookup[shader_hash]);
	}

	LOG_INFO("Loding Shader {}", desc.filename);

	std::string spv_path = "bin/Shaders/" + std::to_string(shader_hash) + ".spv";

	std::vector<uint32_t> spirv;

	if (Path::GetInstance().IsExist(spv_path))
	{
		ShaderDesc spirv_desc = desc;
		spirv_desc.filename   = spv_path;
		spirv_desc.type       = ShaderType::SPIRV;
		spirv                 = ShaderCompiler::GetInstance().Compile(spirv_desc);
	}
	else
	{
		spirv = ShaderCompiler::GetInstance().Compile(desc);

		std::vector<uint8_t> write_data(spirv.size() * 4);
		std::memcpy(write_data.data(), spirv.data(), write_data.size());

		if (!Path::GetInstance().IsExist("bin/Shaders/"))
		{
			Path::GetInstance().CreatePath("bin/Shaders/");
		}

		if (!write_data.empty())
		{
			Path::GetInstance().Save(spv_path, write_data, true);
		}
	}

	m_reflection_data.emplace_back(std::move(ShaderReflection::GetInstance().Reflect(spirv, desc.stage)));

	VkShaderModuleCreateInfo shader_module_create_info = {};
	shader_module_create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_create_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_create_info.pCode                    = spirv.data();
	VkShaderModule shader_module;
	vkCreateShaderModule(p_device->GetDevice(), &shader_module_create_info, nullptr, &shader_module);

	m_shader_modules.push_back(shader_module);
	m_lookup[shader_hash]    = m_shader_modules.size() - 1;
	m_mapping[shader_module] = m_shader_modules.size() - 1;

	return m_shader_modules.at(m_lookup[shader_hash]);
}

const ShaderReflectionData &ShaderAllocator::Reflect(VkShaderModule shader)
{
	ASSERT(m_mapping.find(shader) != m_mapping.end());
	return m_reflection_data.at(m_mapping.at(shader));
}
}        // namespace Ilum