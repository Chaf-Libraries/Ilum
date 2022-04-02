#include "Shader.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/PhysicalDevice.hpp"

#include "Engine/Context.hpp"
#include "Engine/Engine.hpp"

#include "File/FileSystem.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "ShaderCache.hpp"
#include "ShaderCompiler.hpp"
#include "ShaderReflection.hpp"

namespace Ilum
{
Shader &Shader::load(const std::string &filename, VkShaderStageFlagBits stage, Type type, const std::string &entry_point)
{
	if (!FileSystem::isExist("Shader/"))
	{
		FileSystem::createPath("Shader/");
	}

	std::string spv_path = "Shader/" + FileSystem::getFileName(filename, false) + "_" + std::to_string(stage) + ".spv";

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (FileSystem::isExist(spv_path))
	{
		shader_module = GraphicsContext::instance()->getShaderCache().load(spv_path, stage, Type::SPIRV, entry_point);
	}
	else
	{
		shader_module = GraphicsContext::instance()->getShaderCache().load(filename, stage, type, entry_point);
	}

	if (!shader_module)
	{
		VK_ERROR("Failed to load shader: {}", filename);
	}

	VK_Debugger::setName(shader_module, FileSystem::getFileName(filename).c_str());

	m_shader_modules[stage].emplace_back(shader_module);
	m_relection_data += GraphicsContext::instance()->getShaderCache().reflect(shader_module);
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
	else if ((m_stage & VK_SHADER_STAGE_VERTEX_BIT))
	{
		return VK_PIPELINE_BIND_POINT_GRAPHICS;
	}
	else if (m_stage & VK_SHADER_STAGE_RAYGEN_BIT_KHR)
	{
		return VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR;
	}

	return VK_PIPELINE_BIND_POINT_MAX_ENUM;
}

const std::unordered_map<VkShaderStageFlagBits, std::vector<VkShaderModule>> &Shader::getShaders() const
{
	return m_shader_modules;
}
}        // namespace Ilum