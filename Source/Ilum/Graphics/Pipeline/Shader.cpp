#include "Shader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"

#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "File/FileSystem.hpp"

#include "Graphics/GraphicsContext.hpp"

#include <glslang/Include/ResourceLimits.h>
#include <SPIRV/GLSL.std.450.h>
#include <SPIRV/GlslangToSpv.h>

#include <spirv_glsl.hpp>

#include "ShaderCompiler.hpp"
#include "ShaderReflection.hpp"

namespace Ilum
{
Shader::~Shader()
{
	for (auto &[stage, shader_module] : m_shader_module_cache)
	{
		if (shader_module != VK_NULL_HANDLE)
		{
			vkDestroyShaderModule(Engine::instance()->getContext().getSubsystem<GraphicsContext>()->getLogicalDevice(), shader_module, nullptr);
		}
	}

	m_shader_module_cache.clear();
}

Shader &Shader::load(const std::string &filename, VkShaderStageFlagBits stage, Type type)
{
	std::vector<uint8_t> raw_data;
	FileSystem::read(filename, raw_data, type == Type::SPIRV);

	auto spirv = ShaderCompiler::compile(raw_data, stage, type);

	m_relection_data += ShaderReflection::reflect(spirv, stage);

	VkShaderModuleCreateInfo shader_module_create_info = {};
	shader_module_create_info.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_create_info.codeSize                 = spirv.size() * sizeof(uint32_t);
	shader_module_create_info.pCode                    = spirv.data();

	VkShaderModule shader_module;
	if (!VK_CHECK(vkCreateShaderModule(GraphicsContext::instance()->getLogicalDevice(), &shader_module_create_info, nullptr, &shader_module)))
	{
		VK_ERROR("Failed to create shader module");
		return *this;
	}

	m_shader_module_cache.emplace(stage, shader_module);
	m_stage |= stage;

	return *this;
}

const ReflectionData &Shader::getReflectionData() const
{
	return m_relection_data;
}

std::vector<VkPushConstantRange> Shader::getPushConstantRanges() const
{

	std::vector<VkPushConstantRange> push_constant_ranges;
	for (auto &constant : m_relection_data.constants)
	{
		if (constant.type == ReflectionData::Constant::Type::Push)
		{
			VkPushConstantRange push_constant_range = {};
			push_constant_range.stageFlags          = constant.stage;
			push_constant_range.size                = constant.size;
			push_constant_range.offset              = constant.offset;
			push_constant_ranges.push_back(push_constant_range);
		}
	}

	return push_constant_ranges;
}

VkPipelineBindPoint Shader::getBindPoint() const
{
	if (m_stage & VK_SHADER_STAGE_COMPUTE_BIT)
	{
		return VK_PIPELINE_BIND_POINT_COMPUTE;
	}
	else if ((m_stage & VK_SHADER_STAGE_VERTEX_BIT) && (m_stage & VK_SHADER_STAGE_FRAGMENT_BIT))
	{
		return VK_PIPELINE_BIND_POINT_GRAPHICS;
	}

	return VK_PIPELINE_BIND_POINT_MAX_ENUM;
}

const std::unordered_map<VkShaderStageFlagBits, VkShaderModule> &Shader::getShaders() const
{
	return m_shader_module_cache;
}
}        // namespace Ilum