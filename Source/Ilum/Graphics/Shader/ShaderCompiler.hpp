#pragma once

#include "Shader.hpp"

namespace Ilum
{
class ShaderCompiler
{
  public:
	static void init();

	static void destroy();

	static std::vector<uint32_t> compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlagBits stage, Shader::Type type, const std::string &entry_point = "main", const std::vector<std::string> &macros = {});
};
}        // namespace Ilum