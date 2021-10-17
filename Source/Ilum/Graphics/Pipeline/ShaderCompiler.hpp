#pragma once

#include "Shader.hpp"

namespace Ilum
{
	class ShaderCompiler
	{
	  public:
	    static std::vector<uint32_t> compile(const std::vector<uint8_t> &data, VkShaderStageFlags stage, Shader::Type type);
	};
}