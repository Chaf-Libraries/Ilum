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

	Shader &load(const std::string &filename, VkShaderStageFlagBits stage, Type type);

	const ReflectionData &getReflectionData() const;

	std::vector<VkPushConstantRange> getPushConstantRanges() const;

	VkPipelineBindPoint getBindPoint() const;

	const std::unordered_map<VkShaderStageFlagBits, std::vector<VkShaderModule>> &getShaders() const;

  private:
	VkShaderStageFlags m_stage = 0;

	ReflectionData m_relection_data;

	std::unordered_map<VkShaderStageFlagBits, std::vector<VkShaderModule>> m_shader_modules;


};
}        // namespace Ilum