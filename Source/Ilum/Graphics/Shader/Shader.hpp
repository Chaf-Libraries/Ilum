#pragma once

#include "Utils/PCH.hpp"

#include "ShaderReflection.hpp"

namespace Ilum
{
class LogicalDevice;

class Shader
{
  public:
	enum class Type
	{
		GLSL,
		HLSL,
		SPIRV
	};

  public:
	Shader() = default;

	~Shader() = default;

	Shader &load(const std::string &filename, VkShaderStageFlagBits stage, Type type, const std::string &entry_point = "main", const std::vector<std::string> &macros = {});

	const ReflectionData &getReflectionData() const;

	std::vector<VkPushConstantRange> getPushConstantRanges() const;

	VkPipelineBindPoint getBindPoint() const;

	const std::unordered_map<VkShaderStageFlagBits, std::vector<std::pair<VkShaderModule, std::string>>> &getShaders() const;

  private:
	VkShaderStageFlags m_stage = 0;

	ReflectionData m_relection_data;

	std::unordered_map<VkShaderStageFlagBits, std::vector<std::pair<VkShaderModule, std::string>>> m_shader_modules;	// shader module - entry point
};
}        // namespace Ilum