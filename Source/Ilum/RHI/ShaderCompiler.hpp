#pragma once

#include <Core/Hash.hpp>
#include <Core/Singleton.hpp>

#include <volk.h>

#include <string>
#include <vector>

namespace Ilum
{
enum class ShaderType
{
	GLSL,
	HLSL,
	SPIRV
};

struct ShaderDesc
{
	std::string              filename;
	VkShaderStageFlagBits    stage;
	ShaderType               type;
	std::string              entry_point = "main";
	std::vector<std::string> macros      = {};

	size_t Hash() const
	{
		size_t hash = 0;
		HashCombine(hash, filename);
		HashCombine(hash, stage);
		HashCombine(hash, type);
		HashCombine(hash, entry_point);
		for (auto &macro : macros)
		{
			HashCombine(hash, macro);
		}
		return hash;
	}
};

class ShaderCompiler : public Singleton<ShaderCompiler>
{
  public:
	ShaderCompiler();
	~ShaderCompiler();

	std::vector<uint32_t> Compile(const ShaderDesc &desc);
};
}        // namespace Ilum