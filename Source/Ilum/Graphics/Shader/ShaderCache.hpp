#pragma once

#include "Utils/PCH.hpp"

#include "Shader.hpp"
#include "ShaderReflection.hpp"

namespace Ilum
{
class ShaderCache
{
  public:
	ShaderCache() = default;

	~ShaderCache();

	VkShaderModule load(const std::string &filename, VkShaderStageFlagBits stage, Shader::Type type = Shader::Type::GLSL, const std::string &entry_point = "main");

	VkShaderModule getShader(const std::string &filename);

	const ReflectionData &reflect(VkShaderModule shader);

	const std::unordered_map<std::string, size_t> &getShaders() const;

  private:
	std::vector<VkShaderModule> m_shader_modules;
	std::vector<ReflectionData> m_reflection_data;

	std::unordered_map<std::string, size_t>    m_lookup;
	std::unordered_map<VkShaderModule, size_t> m_mapping;
};
}        // namespace Ilum