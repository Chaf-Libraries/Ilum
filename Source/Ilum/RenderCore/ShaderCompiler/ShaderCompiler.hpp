#pragma once

#include <Core/Singleton.hpp>

#include <RHI/RHIDefinitions.hpp>
#include <RHI/RHIShader.hpp>

#include <string>
#include <vector>

namespace Ilum
{
enum class ShaderSource
{
	HLSL,
	GLSL
};

enum class ShaderTarget
{
	SPIRV,
	DXIL
};

struct ShaderDesc
{
	std::string              code;
	ShaderSource             source;
	ShaderTarget             target;
	RHIShaderStage           stage;
	std::string              entry_point;
	std::vector<std::string> macros = {};
};

class ShaderCompiler : public Singleton<ShaderCompiler>
{
  public:
	ShaderCompiler();

	~ShaderCompiler();

	std::vector<uint8_t> Compile(const ShaderDesc &desc, ShaderMeta& meta);
};

}        // namespace Ilum