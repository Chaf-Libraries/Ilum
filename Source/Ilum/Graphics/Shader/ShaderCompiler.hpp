#pragma once

#include "Shader.hpp"

namespace Ilum
{
	class ShaderCompiler
	{
	  public:
	    static std::vector<uint32_t> compile(const std::string &filename, const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type, const std::string &entry_point = "main");
	};
}